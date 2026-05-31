# Demo live pentru dizertatie

Acest repo a fost adaptat pentru un demo live bazat pe taskurile din repository-ul dizertatiei:

- intentie principala;
- status final al conversatiei;
- detectarea neconcordantelor.

Demo-ul pastreaza o conversatie live cu utilizatorul, salveaza intern transcriptul `USER` / `ASSISTANT`, iar la final returneaza un JSON cu aceeasi structura conceptuala ca experimentele din dizertatie.

## Rulare

Din directorul proiectului:

```bash
python live_banking_demo.py
```

## Interfata web

Demo-ul are si o interfata web pentru prezentare. Asistentul se numeste **Banutel** in cod si **Bănuțel** in interfata.

```bash
python web_demo_server.py
```

Apoi deschide:

```text
http://127.0.0.1:8787
```

Pentru voce Zevo TTS in interfata, seteaza cheia:

```powershell
$env:ZEVO_API_KEY="cheia_ta_zevo"
python web_demo_server.py
```

Pentru a afisa in interfata optiunea de apel:

```powershell
$env:ZEVO_PHONE_NUMBER="+407xxxxxxxx"
$env:PUBLIC_WEBHOOK_URL="https://domeniul-tau-public/api/telephony/inbound"
python web_demo_server.py
```

Interfata include:

- chat live cu Bănuțel;
- mod text;
- mod voce cu buton de microfon, Zevo STT si transcript;
- mod telefon cu numar de apel afisat;
- knowledge base local construit din cele 180 de conversatii;
- afisarea exemplelor similare folosite ca referinta;
- redare TTS pentru raspunsurile botului;
- golire cache pentru fisierele audio generate;
- endpoint local pentru integrare telefonica.

## Knowledge base conversațional

Fisierul `conversation_kb.py` incarca datasetul:

```text
master_dataset_refined_180.json
```

Pentru fiecare mesaj live, sistemul cauta conversatii similare in cele 180 de exemple si foloseste rezultatul pentru:

- sugerarea intentiei initiale;
- explicarea demo-ului prin exemple similare;
- afisarea in interfata a conversatiilor de referinta.

Retrieval-ul este local si explicabil: textul conversatiilor este normalizat, transformat in tokeni si comparat lexical prin similaritate cosinus.

Pentru a vedea si transcriptul inainte de analiza:

```bash
python live_banking_demo.py --show-transcript
```

In timpul conversatiei, scrie `analiza`, `final`, `stop` sau `exit` ca sa inchei dialogul si sa afisezi evaluarea finala.

## Exemplu rapid

```text
USER: Mi-am pierdut cardul si vreau sa il blochez.
VOICEBOT: Imi pare rau pentru situatie. Pentru siguranta, spuneti ultimele 4 cifre ale cardului pe care doriti sa il blocati.
USER: 1234
VOICEBOT: Confirmati blocarea cardului terminat in 1234? Raspundeti cu da sau nu.
USER: da
VOICEBOT: Am blocat cardul terminat in 1234. Veti primi o confirmare prin SMS.
USER: analiza
```

Output-ul final va include:

```json
{
  "intent": {
    "intent": "blocare_card",
    "confidence": "high",
    "reasoning": "..."
  },
  "final_status": {
    "final_status": "rezolvata",
    "confidence": "high",
    "reasoning": "..."
  },
  "incongruities": {
    "has_incongruity": false,
    "incongruity_type": null,
    "confidence": "medium",
    "reasoning": "..."
  }
}
```

## Fisiere adaugate

- `live_banking_demo.py` - demo-ul live de conversatie bancara.
- `conversation_evaluator.py` - evaluator offline pentru intentie, status final si neconcordante.

## Legatura cu proiectul vechi

Fisierele originale `chatbot.py`, `zevo_stt.py` si `zevo_tts.py` au fost pastrate. Noul demo foloseste input text din terminal, ca sa poata fi rulat usor la prezentare fara dependente audio sau chei API. Daca vrei ulterior demo vocal complet, functiile Zevo STT/TTS existente pot fi conectate la `BankingVoicebotDemo.handle_user_message()`.

## Integrare telefonica

Serverul web expune endpoint-ul:

```text
POST /api/telephony/inbound
```

Payload acceptat:

```json
{
  "call_id": "apel-001",
  "transcript": "Mi-am pierdut cardul si vreau sa il blochez"
}
```

Raspuns:

```json
{
  "reply_text": "Îmi pare rău pentru situație...",
  "transcript": []
}
```

Pentru ca un numar real sa sune in acest endpoint trebuie configurat webhook-ul in serviciul telefonic Zevo sau in gateway-ul SIP/telefonie folosit. In lipsa credentialelor si a URL-ului public de webhook, proiectul ofera doar endpoint-ul local si logica de raspuns.
