import argparse
import json
import os
import re
import sys
import wave
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from conversation_evaluator import ConversationEvaluator, Turn, dump_evaluation, format_conversation
from speech_normalizer import normalize_for_tts
from zevo_tts import (
    DEFAULT_AUDIO_FORMAT_TTS,
    DEFAULT_BITS_PER_SAMPLE_TTS,
    DEFAULT_SAMPLE_RATE_TTS,
    TTSRequestParams,
    ZEVO_TTS_API_URI,
    text_to_speech_ws,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


DEFAULT_DATASET = (
    Path(__file__).resolve().parent.parent
    / "Sistem-de-monitorizare-a-interac-iunilor-voicebotilor-folosind-modele-lingvistice-mari-LLM-"
    / "data"
    / "master_dataset_refined_180.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ruleaza un subset din dataset prin evaluator si optional prin Zevo TTS.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Calea catre master_dataset_refined_180.json.")
    parser.add_argument("--limit", type=int, default=5, help="Numarul de conversatii de rulat.")
    parser.add_argument("--offset", type=int, default=0, help="De la ce conversatie incepe subsetul.")
    parser.add_argument("--intent", default=None, help="Filtreaza dupa intentia din dataset.")
    parser.add_argument("--tts", action="store_true", help="Genereaza fisiere WAV cu Zevo TTS pentru replicile din subset.")
    parser.add_argument("--tts-key", default=os.getenv("ZEVO_API_KEY"), help="Cheia Zevo. Alternativ, seteaza ZEVO_API_KEY.")
    parser.add_argument("--voice", default=os.getenv("ZEVO_TTS_VOICE", "gia"), help="Vocea Zevo TTS.")
    parser.add_argument("--output-dir", default="tts_subset_output", help="Directorul pentru fisiere WAV si raport.")
    args = parser.parse_args()

    conversations = load_subset(Path(args.dataset), args.limit, args.offset, args.intent)
    evaluator = ConversationEvaluator()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report: List[Dict[str, object]] = []
    for index, conversation in enumerate(conversations, start=1):
        conv_id = conversation["conversation_id"]
        turns = conversation["turns"]
        print(f"\n=== {index}. {conv_id} ===")
        print(f"Dataset: intent={conversation.get('intent')} final_status={conversation.get('final_status')}")
        print(format_conversation(turns))

        quality_notes = pronunciation_notes(turns)
        if quality_notes:
            print("Observatii pronuntie/text:")
            for note in quality_notes:
                print(f"- {note}")
        else:
            print("Observatii pronuntie/text: diacriticele si textul arata bine.")

        evaluation = evaluator.evaluate(turns)
        print("Evaluare demo:")
        print(dump_evaluation(evaluation))

        audio_files: List[str] = []
        if args.tts:
            if not args.tts_key:
                raise SystemExit("Lipseste cheia Zevo. Seteaza ZEVO_API_KEY sau foloseste --tts-key.")
            audio_files = synthesize_conversation(turns, conv_id, output_dir, args.tts_key, args.voice)
            print("Audio generat:")
            for audio_file in audio_files:
                print(f"- {audio_file}")

        report.append(
            {
                "conversation_id": conv_id,
                "dataset_intent": conversation.get("intent"),
                "dataset_final_status": conversation.get("final_status"),
                "quality_notes": quality_notes,
                "evaluation": evaluation.to_dict(),
                "audio_files": audio_files,
            }
        )

    report_path = output_dir / "subset_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRaport scris in: {report_path}")


def load_subset(dataset_path: Path, limit: int, offset: int, intent: Optional[str]) -> List[Dict[str, object]]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    conversations = data["conversations"]
    if intent:
        conversations = [conv for conv in conversations if conv.get("intent") == intent]
    return conversations[offset : offset + limit]


def pronunciation_notes(turns: Iterable[Turn]) -> List[str]:
    notes: List[str] = []
    mojibake_pattern = re.compile(r"[ÃÄÅ][^\s]*|È[^\s]*")
    for idx, turn in enumerate(turns, start=1):
        text = turn.get("text", "")
        if mojibake_pattern.search(text):
            notes.append(f"Turn {idx} pare sa contina text cu encoding stricat: {text[:80]}")
        if "voicebot" in text.lower():
            notes.append(f"Turn {idx} contine termenul englezesc 'voicebot', care poate fi pronuntat nefiresc in romana.")
    return notes


def synthesize_conversation(turns: Iterable[Turn], conv_id: str, output_dir: Path, key: str, voice: str) -> List[str]:
    import asyncio

    async def synthesize_all() -> List[str]:
        generated: List[str] = []
        for turn_index, turn in enumerate(turns, start=1):
            text = normalize_for_tts(turn.get("text", "").strip())
            if not text:
                continue
            safe_role = turn.get("role", "turn")
            filename = output_dir / f"{conv_id}_{turn_index:02d}_{safe_role}.wav"
            params = TTSRequestParams(
                key=key,
                text=text,
                voice=voice,
                audio_format=DEFAULT_AUDIO_FORMAT_TTS,
                sample_rate=DEFAULT_SAMPLE_RATE_TTS,
                bits_per_sample=DEFAULT_BITS_PER_SAMPLE_TTS,
            )
            audio_data = await text_to_speech_ws(ZEVO_TTS_API_URI, params)
            if not audio_data:
                raise RuntimeError(f"Zevo TTS nu a returnat audio pentru {conv_id}, turn {turn_index}.")
            write_wav(filename, audio_data, DEFAULT_SAMPLE_RATE_TTS, DEFAULT_BITS_PER_SAMPLE_TTS)
            generated.append(str(filename))
        return generated

    return asyncio.run(synthesize_all())


def write_wav(path: Path, audio_data: bytes, sample_rate: int, bits_per_sample: int) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(bits_per_sample // 8)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)


if __name__ == "__main__":
    main()
