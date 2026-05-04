import json
from openai import AzureOpenAI
from datetime import datetime, timedelta
import os

# Importuri pentru STT si TTS
from zevo_stt import record_and_transcribe_from_mic, DEFAULT_DOMAIN_STT_GENERAL 
from zevo_tts import speak_text_zevo 

# --- Configurare și Constante ---
CONFIG = {
    "API_KEY": "c5fa3b989ae1483dbc1f071630926db6", 
    "AZURE_ENDPOINT": "https://laborator-lucian.openai.azure.com",
    "MODEL": "laborator-lucian",
    "API_VERSION": "2023-10-01-preview"
}
ZEVO_API_KEY = "icvsilab2025" 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(SCRIPT_DIR, "knowledge_base.json")
DIALOGUES_PATH = os.path.join(SCRIPT_DIR, "dialogues.txt")

MAX_RETRIES = 2 
VALID_CITIES = ["bucurești", "cluj-napoca", "iași", "timișoara", "constanța", "brașov", "sibiu", "oradea", "arad", "ploiești"]
DISPLAY_CITIES = {
    "bucurești": "București", "cluj-napoca": "Cluj-Napoca", "iași": "Iași", 
    "timișoara": "Timișoara", "constanța": "Constanța", "brașov": "Brașov", 
    "sibiu": "Sibiu", "oradea": "Oradea", "arad": "Arad", "ploiești": "Ploiești"
}

# --- Client Azure OpenAI ---
client = AzureOpenAI(
    api_key=CONFIG["API_KEY"],
    api_version=CONFIG["API_VERSION"],
    azure_endpoint=CONFIG["AZURE_ENDPOINT"]
)

# --- Clasa NLPEngine ---
class NLPEngine:
    def __init__(self, gpt_client, kb_manager):
        self.gpt_client = gpt_client
        self.kb_manager = kb_manager 

    def _call_gpt(self, system_prompt, user_input):
        try:
            response = self.gpt_client.chat.completions.create(
                model=CONFIG["MODEL"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.0,
                n=1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"({datetime.now().strftime('%H:%M:%S')}) EROARE API GPT: {e}")
            return None

    def extract_and_normalize_name(self, user_input_name_phrase):
        if not user_input_name_phrase:
            return None
        prompt = (
            "Utilizatorul își spune numele. Extrage numele complet (prenume și nume de familie) din următoarea frază. "
            "Ignoră cuvintele de politețe precum 'bună ziua', 'numele meu este', etc. "
            "Dacă sunt furnizate mai multe nume/prenume, încearcă să le păstrezi. "
            "Încearcă să returnezi numele în formatul 'Prenume Nume_Familie'. "
            "Dacă ordinea este inversată (ex: 'Mârzan Teodora'), corecteaz-o în 'Teodora Mârzan'. "
            "Dacă fraza nu pare să conțină un nume clar, returnează null.\n"
            "Exemple:\n"
            "- Input: 'Bună ziua, numele meu este Teodora Mârzan.' -> Output: Teodora Mârzan\n"
            "- Input: 'Popescu Ion Andrei' -> Output: Popescu Ion Andrei (sau Ion Andrei Popescu)\n"
            "- Input: 'mă numesc Vasilescu Ana Maria' -> Output: Ana Maria Vasilescu\n"
            f"Fraza utilizatorului: '{user_input_name_phrase}'\n"
            "Returnează STRICT JSON: {\"full_name\": \"NumeleCompletExtras\" | null}"
        )
        result_str = self._call_gpt(prompt, user_input_name_phrase)
        if result_str is None: return None
        try:
            data = json.loads(result_str)
            full_name = data.get("full_name")
            if full_name and len(full_name.split()) >= 1: # Acceptă și un singur cuvânt ca nume/prenume
                return full_name.strip()
            return None
        except json.JSONDecodeError:
            print(f"({datetime.now().strftime('%H:%M:%S')}) DEBUG: JSONDecodeError în extract_and_normalize_name: '{result_str}'")
            # Ca fallback, încercăm să curățăm inputul de fraze comune
            cleaned_name = user_input_name_phrase.lower()
            phrases_to_remove = ["bună ziua", "numele meu este", "mă numesc", "eu sunt", "sunt"]
            for phrase in phrases_to_remove:
                cleaned_name = cleaned_name.replace(phrase, "")
            cleaned_name = cleaned_name.strip()
            return cleaned_name if len(cleaned_name.split()) >= 1 else None


    def parse_intent(self, user_input):
        valid_cities_display_str = ", ".join(DISPLAY_CITIES.values())
        prompt = (
            f"Ești un asistent pentru clinica Sunset. Utilizatorul poate cere o programare, anulare sau încheiere. "
            f"Orașele deservite sunt: {valid_cities_display_str}.\n"
            "Input utilizator: '{user_input}'.\n"
            "Extrage intenția principală. Dacă se menționează un oraș pentru programare, extrage DOAR numele orașului (ex: din 'programare in Bucuresti', extrage 'Bucuresti'). "
            "Dacă orașul extras nu este în lista orașelor deservite (chiar și cu diacritice), returnează null pentru oraș. "
            "Returnează STRICT JSON: "
            '{"intent": "programare noua" | "anulare" | "incheiere" | "necunoscut", '
            '"city": "numele orașului extras" | null, "specialty": "specializare extrasă" | null, "date": "dată extrasă" | null}'
        )
        result_str = self._call_gpt(prompt, user_input)
        if result_str is None: return None 
        try:
            data = json.loads(result_str)
            extracted_city_raw = data.get("city")
            if extracted_city_raw:
                cleaned_city_name = extracted_city_raw.lower() 
                common_phrases_to_remove = ["in ", " orașul", " aveti", "?"]
                for phrase in common_phrases_to_remove:
                    cleaned_city_name = cleaned_city_name.replace(phrase, "")
                cleaned_city_name = cleaned_city_name.strip()

                if cleaned_city_name in VALID_CITIES:
                    data["city"] = cleaned_city_name 
                else:
                    found_valid_city_in_input = None
                    for valid_city_name_kb in VALID_CITIES: 
                        if valid_city_name_kb in user_input.lower(): 
                            found_valid_city_in_input = valid_city_name_kb
                            break
                    data["city"] = found_valid_city_in_input 
            return data
        except json.JSONDecodeError:
            print(f"({datetime.now().strftime('%H:%M:%S')}) DEBUG: JSONDecodeError în parse_intent pentru: '{result_str}'")
            return {"intent": "necunoscut", "city": None, "specialty": None, "date": None}


    def extract_city_from_input(self, user_input_city_prompt):
        valid_cities_display_str = ", ".join(DISPLAY_CITIES.values())
        prompt = (
            f"Utilizatorul a fost întrebat în ce oraș dorește programarea. Orașele noastre sunt: {valid_cities_display_str}.\n"
            f"Răspunsul utilizatorului este: '{user_input_city_prompt}'.\n"
            "Extrage DOAR numele relevant al orașului din răspunsul utilizatorului. De exemplu:\n"
            "- Din 'la valcea nu aveti?', extrage 'Vâlcea'.\n" 
            "- Din 'Bucuresti, va rog.', extrage 'București'.\n"
            "- Din 'Unde aveti clinica? In Slatina?', extrage 'Slatina'.\n"
            "Dacă răspunsul este neclar, ambiguu sau nu pare a conține un nume de oraș, returnează null.\n"
            "Returnează STRICT JSON: {\"city_extracted\": \"numele orașului extras cu diacritice corecte\" | null}"
        )
        result_str = self._call_gpt(prompt, user_input_city_prompt)
        if result_str is None: return None
        try:
            data = json.loads(result_str)
            extracted_city = data.get("city_extracted")
            if extracted_city: 
                return extracted_city.strip().replace("?", "").lower() 
            return None
        except json.JSONDecodeError:
            print(f"({datetime.now().strftime('%H:%M:%S')}) DEBUG: JSONDecodeError în extract_city_from_input: '{result_str}'")
            return None

    def parse_follow_up_intent(self, user_input):
        prompt = ("Returnează STRICT JSON: {\"intent\":\"programare noua\"|\"anulare\"|\"incheiere\"|\"necunoscut\"}.")
        result_str = self._call_gpt(prompt, user_input)
        if result_str is None: return None
        try:
            return json.loads(result_str).get("intent", "necunoscut")
        except (json.JSONDecodeError, KeyError):
            return "necunoscut"

    def normalize_date_input(self, user_input_date, available_dates, context_preference=""):
        safe_context_preference = context_preference or "" 
        safe_user_input_date = user_input_date or ""   

        if not safe_user_input_date and not safe_context_preference: return None
        if not available_dates: return "no_dates_for_specialty"

        current_date_str = datetime.now().strftime("%d.%m.%Y")
        try:
            sorted_available_dates = sorted(available_dates, key=lambda d: datetime.strptime(d, "%d.%m.%Y"))
        except ValueError:
            print(f"({datetime.now().strftime('%H:%M:%S')}) DEBUG: Eroare la sortarea datelor în normalize_date_input.")
            sorted_available_dates = available_dates 

        available_dates_str = "\n".join(f"- {d}" for d in sorted_available_dates)
        
        user_query_part = f"Utilizatorul a zis: '{safe_user_input_date or safe_context_preference}'."
        if safe_user_input_date and safe_context_preference: 
             user_query_part = f"Utilizatorul a menționat '{safe_user_input_date}' și ar prefera '{safe_context_preference}'."

        prompt = (
            f"Data curentă: {current_date_str} (anul este {datetime.now().year}).\n"
            "Următoarele date sunt disponibile pentru programare (format strict dd.mm.yyyy), în ordine cronologică:\n"
            f"{available_dates_str}\n\n"
            f"{user_query_part}\nAnalizează inputul utilizatorului (care poate fi 'azi', 'mâine', 'luni', 'săptămâna viitoare marți', '10 iunie', 'pe zece', 'cât mai repede', 'prima liberă', 'cât de curând posibil'). " 
            "Returnează STRICT o dată din listă sau 'data indisponibila'."
        )
        final_input_for_gpt = safe_user_input_date if safe_user_input_date else safe_context_preference
        result = self._call_gpt(prompt, final_input_for_gpt)
        
        if result is None: return None 

        if result.lower() == "data indisponibila":
            if any(k in safe_context_preference.lower() for k in ["curand", "prima", "repede"]) and sorted_available_dates:
                return sorted_available_dates[0]
            return None
        
        return result if result in sorted_available_dates else None

    def resolve_specialty_from_symptom(self, user_input):
        if not user_input: return None
        available_specialties = self.kb_manager.get_all_specialties() 
        specialties_str = ", ".join(available_specialties)
        
        prompt = (
            f"Specializări disponibile: {specialties_str}.\nUtilizator: '{user_input}'.\n"
            "Identifică specializarea din listă. Poate fi simptom ('durere inimă'), tip doctor ('de ochi'), sau nume specializare (chiar și parțial 'cardio').\n"
            "Returnează STRICT numele exact al specializării din listă (cu diacriticele din listă) sau 'necunoscut'."
        )
        result = self._call_gpt(prompt, user_input)
        if result is None: return None

        for spec_kb in available_specialties:
            if result.lower() == spec_kb.lower(): 
                return spec_kb 
        
        if result.lower() == "necunoscut": 
            normalized_input = user_input.lower() 
            for spec_kb in available_specialties: 
                if normalized_input in spec_kb.lower() or \
                   (spec_kb.lower() in normalized_input and len(spec_kb) > 3 and len(normalized_input) > 3) : 
                    return spec_kb
            return None
        
        print(f"({datetime.now().strftime('%H:%M:%S')}) DEBUG: GPT a returnat '{result}' care nu se potrivește exact cu {available_specialties}")
        return None


    def parse_confirmation(self, user_input):
        prompt = ("Răspuns utilizator la o întrebare da/nu: '{user_input}'. "
                  "Returnează STRICT JSON: {\"confirmation\": \"da\" | \"nu\" | \"neclar\"}")
        result_str = self._call_gpt(prompt, user_input)
        if result_str is None: return None
        try:
            return json.loads(result_str).get("confirmation", "neclar")
        except (json.JSONDecodeError, KeyError):
            return "neclar"

    def identify_appointment_from_description(self, user_description, appointments_list):
        if not user_description or not appointments_list: return None
        app_list_str = "\n".join([f"Prog {i+1}: {a['specialty']} pe {a['date']} în {DISPLAY_CITIES.get(a['city'], a['city'])}" for i, a in enumerate(appointments_list)])
        prompt = (
            f"Programări:\n{app_list_str}\nUtilizator: '{user_description}'.\n"
            "Identifică la ce programare (index 1-based) se referă (ex: 'prima', 'cea de la cardiologie'). Returnează STRICT indexul sau '0' dacă neclar."
        )
        result_str = self._call_gpt(prompt, user_description)
        if result_str is None: return None
        try:
            idx = int(result_str)
            return appointments_list[idx - 1] if 1 <= idx <= len(appointments_list) else None
        except (ValueError, TypeError, IndexError):
            return None

# --- Clasa KnowledgeBaseManager ---
class KnowledgeBaseManager:
    def __init__(self, kb_path):
        self.kb_path = kb_path
        self.kb = self._load_kb()

    def _load_kb(self):
        try:
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"({datetime.now().strftime('%H:%M:%S')}) EROARE la încărcarea KB '{self.kb_path}': {e}")
            return {"booked_dates": [], "ExempluSpecializare": {"available_dates": []}} 

    def _save_kb(self):
        try:
            with open(self.kb_path, "w", encoding="utf-8") as f:
                json.dump(self.kb, f, ensure_ascii=False, indent=4) 
        except Exception as e: 
            print(f"({datetime.now().strftime('%H:%M:%S')}) EROARE la salvarea KB: {e}")

    def get_all_specialties(self):
        return [spec for spec in self.kb if spec.lower() != "booked_dates"]

    def get_available_dates(self, specialty_query): 
        key = next((k for k in self.kb if k.lower() == specialty_query.lower()), None)
        return self.kb.get(key, {}).get("available_dates", []) if key else []

    def is_specialty_valid(self, specialty_name_query): 
        return any(s.lower() == specialty_name_query.lower() for s in self.get_all_specialties())

    def book_appointment(self, name, city_normalized, specialty_kb_form, date_to_book):
        if not specialty_kb_form or date_to_book not in self.kb.get(specialty_kb_form,{}).get("available_dates", []): 
            print(f"({datetime.now().strftime('%H:%M:%S')}) DEBUG: Book appointment failed - Specialty Key: {specialty_kb_form}, Date: {date_to_book}")
            return False
        self.kb[specialty_kb_form]["available_dates"].remove(date_to_book)
        self.kb.setdefault("booked_dates", []).append({
            "name": name.lower(), "city": city_normalized, 
            "specialty": specialty_kb_form, "date": date_to_book
        })
        self._save_kb()
        return True

    def get_patient_appointments(self, name): 
        return [a for a in self.kb.get("booked_dates", []) if a.get("name", "") == name]

    def cancel_appointment(self, appt_to_cancel): 
        booked_dates = self.kb.get("booked_dates", [])
        if appt_to_cancel not in booked_dates: return False
        
        booked_dates.remove(appt_to_cancel) 
        self.kb["booked_dates"] = booked_dates 

        specialty_kb_form = appt_to_cancel["specialty"]
        if specialty_kb_form in self.kb: 
            dates = self.kb[specialty_kb_form].setdefault("available_dates", [])
            if appt_to_cancel["date"] not in dates: dates.append(appt_to_cancel["date"])
            try: dates.sort(key=lambda d: datetime.strptime(d, "%d.%m.%Y"))
            except ValueError: pass
        self._save_kb()
        return True

    def is_patient_registered(self, name_for_kb): # name_for_kb este deja "prenume nume" lowercase
        return any(a.get("name","") == name_for_kb for a in self.kb.get("booked_dates", []))
    
    def register_new_patient(self, name, cnp=None, phone=None): return True

# --- Clasa DialogManager ---
class DialogManager:
    def __init__(self, nlp_engine, kb_manager, dialogues_path):
        self.nlp = nlp_engine
        self.kb = kb_manager
        self.dialogues = self._load_dialogues(dialogues_path)
        self.current_patient_name = None 
        self.zevo_stt_domain = DEFAULT_DOMAIN_STT_GENERAL 
        self.zevo_tts_voice = "gia" 

    def _load_dialogues(self, path):
        d = {}
        try:
            with open(path, 'r', encoding='utf-8') as f: 
                for line in f:
                    if "=" in line: d[line.strip().split("=", 1)[0]] = line.strip().split("=", 1)[1]
        except FileNotFoundError:
            print(f"({datetime.now().strftime('%H:%M:%S')}) EROARE Dialoguri: Fișier negăsit: {path}")
            return {"welcome_message": "EROARE Dialoguri. Nume?", "goodbye_message": "Pa.", 
                    "technical_difficulty_prompt": "Problema tehnică. Încercați iar.",
                    "ask_action": "Ce doriți?",
                    "stt_failed_retry": "Nu am înțeles. Puteți repeta, vă rog?",
                    "stt_failed_max_retries": "Din păcate, nu am reușit să înțeleg. Reluăm.",
                    "name_input_unclear": "Nu am înțeles clar numele. Repetați?",
                    "name_still_unclear_goodbye": "Nume neclar. Contactați-ne telefonic."}
        return d

    def _get_dialogue(self, key, **kwargs):
        template = self.dialogues.get(key, f"lipseste_dialog:{key}") 
        try: 
            if 'city' in kwargs and isinstance(kwargs['city'], str):
                kwargs['city'] = DISPLAY_CITIES.get(kwargs['city'], kwargs['city'].title())
            if 'valid_cities' in kwargs and isinstance(kwargs['valid_cities'], str):
                 pass
            return template.format(**kwargs)
        except KeyError as e:
            print(f"({datetime.now().strftime('%H:%M:%S')}) EROARE Dialog: Cheie '{e}' lipsă pentru '{key}'. Kwargs: {kwargs}")
            return f"Eroare mesaj (lipsă {e})"
        except Exception as e_fmt:
            print(f"({datetime.now().strftime('%H:%M:%S')}) EROARE Formatare Dialog pentru '{key}': {e_fmt}. Kwargs: {kwargs}")
            return template 


    def _speak(self, text_key, use_ssml=False, **kwargs):
        dialogue_text = self._get_dialogue(text_key, **kwargs)
        text_for_tts = dialogue_text
        
        # Eliminăm ghilimelele care ar putea fi citite de TTS, cu excepția celor din exemple
        # și a celor din jurul placeholder-elor (care sunt deja rezolvate)
        if not ("ex:" in dialogue_text.lower() or "exemplu:" in dialogue_text.lower() or "de exemplu" in dialogue_text.lower() or "{" in dialogue_text):
            text_for_tts = text_for_tts.replace('"', '').replace("'", "")
        
        # Reformulăm prompturile de confirmare pentru naturalețe
        if "(da/nu)" in text_for_tts: 
            text_for_tts = text_for_tts.replace("(da/nu)", "Puteți răspunde cu da sau nu, vă rog?")
        elif "(da / nu)" in text_for_tts: 
            text_for_tts = text_for_tts.replace("(da / nu)", "Vă rog să confirmați cu da sau nu.")
        
        print("\nCHATBOT: " + dialogue_text) 
        speak_text_zevo(ZEVO_API_KEY, text_for_tts, voice=self.zevo_tts_voice)
        return dialogue_text 

    def _listen(self, stt_domain_override=None, phrases_for_stt=None):
        for i in range(MAX_RETRIES + 1): 
            domain_to_use = stt_domain_override if stt_domain_override else self.zevo_stt_domain
            
            transcribed_text = record_and_transcribe_from_mic(
                ZEVO_API_KEY, domain=domain_to_use, phrases_list=phrases_for_stt,
                listen_timeout=7, phrase_timeout=15 )
            
            if transcribed_text: 
                print(f"UTILIZATOR (STT): {transcribed_text}") 
                return transcribed_text.strip().lower() 
            else: 
                if i < MAX_RETRIES:
                    self._speak('stt_failed_retry') 
                else:
                    self._speak('stt_failed_max_retries') 
                    return "" 
        return "" 

    def _get_user_input_voice(self, prompt_key, stt_domain_override=None, phrases_for_stt=None, **kwargs):
        self._speak(prompt_key, **kwargs)
        return self._listen(stt_domain_override=stt_domain_override, phrases_for_stt=phrases_for_stt)

    def _handle_nlp_failure(self):
        self._speak('technical_difficulty_prompt')
        return False 

    def start(self):
        self._speak('welcome_message') 
        
        name_input_stt = ""
        extracted_name = None
        for _ in range(MAX_RETRIES + 1): # Permite reîncercări pentru extragerea numelui
            name_input_stt = self._listen()
            if not name_input_stt and _ < MAX_RETRIES : self._speak('name_input_unclear'); continue
            if not name_input_stt and _ == MAX_RETRIES : self._speak('name_still_unclear_goodbye'); return

            extracted_name = self.nlp.extract_and_normalize_name(name_input_stt)
            if extracted_name:
                break 
            elif _ < MAX_RETRIES:
                self._speak('name_input_unclear')
            else:
                self._speak('name_still_unclear_goodbye')
                return
        
        if not extracted_name: return # Nu s-a putut obține un nume

        name_for_kb = extracted_name.lower() # Folosim numele normalizat și lowercase pentru KB

        if not self.kb.is_patient_registered(name_for_kb):
            # Afișăm numele extras și normalizat de NLP, cu .title()
            self._speak('name_not_found_for_extracted', name=extracted_name.title()) 
            
            # O singură reîncercare pentru nume dacă nu e găsit, dar a fost extras ceva
            name_input_stt_retry = self._listen()
            if not name_input_stt_retry: # Dacă STT eșuează la reîncercare
                 self._speak('name_still_unclear_goodbye'); return

            extracted_name_retry = self.nlp.extract_and_normalize_name(name_input_stt_retry)
            if not extracted_name_retry:
                self._speak('name_still_unclear_goodbye'); return
            
            name_for_kb = extracted_name_retry.lower() # Actualizăm numele pentru KB

            if not self.kb.is_patient_registered(name_for_kb):
                self._speak('name_still_not_found_for_extracted', name=extracted_name_retry.title()) 
                consent_input_stt = self._get_user_input_voice('ask_registration_consent', phrases_for_stt=["da", "nu", "sigur", "desigur", "da doresc"]) 
                if not consent_input_stt: return

                confirmation = self.nlp.parse_confirmation(consent_input_stt)
                if confirmation is None: return self._handle_nlp_failure()
                if confirmation == 'da':
                    self.current_patient_name = name_for_kb 
                    self.kb.register_new_patient(self.current_patient_name) 
                    self._speak('registration_complete') 
                else:
                    self._speak('goodbye_message'); return
            else:
                self.current_patient_name = name_for_kb
        else:
            self.current_patient_name = name_for_kb
        
        if self.current_patient_name: 
            display_name_at_welcome = extracted_name.title() if extracted_name else self.current_patient_name.title()
            self._speak('welcome_known_patient', name=display_name_at_welcome) 
            self._main_dialog_loop()

    def _main_dialog_loop(self):
        active_session = True
        while active_session:
            user_action_input = self._get_user_input_voice('ask_action') # Prompt scurtat
            if not user_action_input: continue 

            intent_data = self.nlp.parse_intent(user_action_input)
            if intent_data is None: 
                if not self._handle_nlp_failure(): continue
                else: break 
            
            intent = intent_data.get("intent")
            action_processed_successfully = False 

            if intent == "programare noua": action_processed_successfully = self._handle_new_appointment(intent_data) 
            elif intent == "anulare": action_processed_successfully = self._handle_cancellation()
            elif intent == "incheiere": self._speak('goodbye_message'); active_session = False
            else: self._speak('fallback_understanding_prompt'); 
            
            if not active_session: continue 

            follow_up_input = self._get_user_input_voice('follow_up_prompt_generic', 
                                                            phrases_for_stt=["programare", "anulare", "da", "nu", "stop", "gata", "la revedere", "mulțumesc", "altceva"])
            if not follow_up_input: continue 

            follow_up_intent = self.nlp.parse_follow_up_intent(follow_up_input)
            if follow_up_intent is None: 
                if not self._handle_nlp_failure(): continue
                else: break
            
            if follow_up_intent == "programare noua": continue 
            elif follow_up_intent == "anulare": continue 
            elif follow_up_intent == "incheiere": self._speak('goodbye_message'); active_session = False
            else: self._speak('fallback_understanding_prompt_follow_up')


    def _handle_new_appointment(self, initial_data):
        self._speak('start_new_appointment_flow') 
        
        city = initial_data.get("city") 
        city_list_presented_this_turn = False # Flag pentru a controla afișarea listei de orașe

        if not city: 
            valid_cities_for_dialogue = ", ".join(DISPLAY_CITIES.values()) 
            self._speak('ask_city_options_intro', valid_cities=valid_cities_for_dialogue)
            city_list_presented_this_turn = True 

            for i in range(MAX_RETRIES + 1):
                city_input_raw_stt = self._listen(phrases_for_stt=list(VALID_CITIES) + ["vreau in", "doresc la", "programare la"]) 
                
                if not city_input_raw_stt: 
                    if i < MAX_RETRIES: continue 
                    else: self._speak('city_selection_failed_no_input'); return False

                extracted_city_name_nlp = self.nlp.extract_city_from_input(city_input_raw_stt) 
                
                if extracted_city_name_nlp is None and i < MAX_RETRIES: 
                    if not self._handle_nlp_failure(): continue
                    else: return False

                if extracted_city_name_nlp:
                    city_input_lower = extracted_city_name_nlp 
                    if city_input_lower in VALID_CITIES:
                        city = city_input_lower; break 
                    else: 
                        # Orașul menționat de utilizator nu este valid.
                        # Afișăm lista de orașe doar dacă nu a fost deja prezentată în acest ciclu de eroare
                        # sau dacă este prima eroare după promptul inițial.
                        if not city_list_presented_this_turn or i > 0: # i > 0 înseamnă că e o reîncercare după o eroare
                             self._speak('city_not_valid_retry_with_list', 
                                        city_name=DISPLAY_CITIES.get(extracted_city_name_nlp, extracted_city_name_nlp.title()), 
                                        valid_cities=valid_cities_for_dialogue)
                             city_list_presented_this_turn = True 
                        else: 
                             self._speak('city_not_valid_specific', city_name=DISPLAY_CITIES.get(extracted_city_name_nlp, extracted_city_name_nlp.title()))
                        
                        if i == MAX_RETRIES: 
                             self._speak('city_selection_failed_after_retries_no_list')
                             return False
                        # Bucla continuă pentru o nouă încercare, utilizatorul a văzut lista (sau i s-a reamintit)
                        continue 
                else: 
                    if i < MAX_RETRIES: self._speak('city_input_unclear_retry_short') 
                    else: self._speak('city_selection_failed_unclear'); return False
            if not city: return False 
        
        city_display_name = DISPLAY_CITIES.get(city, city.title()) 

        specialty = initial_data.get("specialty") 
        symptom_context_for_dialogue = initial_data.get("specialty","") 
        symptom_provided = False 
        
        if specialty:
            resolved_s_kb = self.nlp.resolve_specialty_from_symptom(specialty) 
            if resolved_s_kb and self.kb.is_specialty_valid(resolved_s_kb):
                # Verificăm dacă inputul original (din intent) NU este direct numele unei specializări
                # pentru a considera că a oferit un simptom.
                if initial_data.get("specialty", "").lower() not in (s.lower() for s in self.kb.get_all_specialties()):
                    symptom_provided = True
                specialty = resolved_s_kb 
            else:
                specialty = None

        if not specialty:
            all_specs_display_str = ", ".join(self.kb.get_all_specialties()) 
            for i in range(MAX_RETRIES + 1):
                s_input_stt = self._get_user_input_voice('ask_specialty_or_symptom', specialties=all_specs_display_str)
                if not s_input_stt: 
                    if i < MAX_RETRIES: continue
                    else: self._speak('specialty_selection_failed'); return False
                
                symptom_context_for_dialogue = s_input_stt 
                resolved_s = self.nlp.resolve_specialty_from_symptom(s_input_stt) 
                if resolved_s is None: 
                    if i < MAX_RETRIES: 
                        if not self._handle_nlp_failure(): continue
                        else: return False 
                elif self.kb.is_specialty_valid(resolved_s): 
                    specialty = resolved_s
                    if s_input_stt.lower() not in (s.lower() for s in self.kb.get_all_specialties()):
                        symptom_provided = True
                    break
                
                if i < MAX_RETRIES: self._speak('specialty_not_recognized_retry', specialties=all_specs_display_str)
                else: self._speak('specialty_selection_failed'); return False
            if not specialty: return False
        
        if symptom_provided:
            # Folosim symptom_context_for_dialogue care conține inputul original al utilizatorului
            self._speak('symptom_based_specialty_suggestion', specialty=specialty, city=city_display_name)
        else:
            self._speak('chosen_specialty_is', specialty=specialty, city=city_display_name)

        available_dates = self.kb.get_available_dates(specialty) 
        if not available_dates: self._speak('no_dates_for_specialty_in_city', specialty=specialty, city=city_display_name); return False

        chosen_date = None
        raw_date_from_intent = initial_data.get("date") 
        date_preference_context = ""

        if raw_date_from_intent: 
            if any(keyword in raw_date_from_intent.lower() for keyword in ["curand", "repede", "prima", "urgent", "posibil"]):
                date_preference_context = raw_date_from_intent
                raw_date_from_intent = "" 
        
        if raw_date_from_intent or date_preference_context:
            normalized_date = self.nlp.normalize_date_input(raw_date_from_intent, available_dates, date_preference_context)
            if normalized_date is None: pass
            elif normalized_date == "no_dates_for_specialty":
                 self._speak('no_dates_for_specialty_in_city', specialty=specialty, city=city_display_name); return False
            elif normalized_date: chosen_date = normalized_date
        
        if not chosen_date:
            for i in range(MAX_RETRIES + 1):
                date_input_stt = self._get_user_input_voice('ask_date_options_natural', specialty=specialty, city=city_display_name, dates=", ".join(available_dates))
                if not date_input_stt: 
                    if i < MAX_RETRIES: continue
                    else: self._speak('date_selection_failed'); return False
                
                current_date_preference_context = "" 
                if date_input_stt: 
                    if any(keyword in date_input_stt.lower() for keyword in ["curand", "repede", "prima", "urgent", "posibil"]):
                         current_date_preference_context = date_input_stt 
                
                normalized_date = self.nlp.normalize_date_input(date_input_stt, available_dates, current_date_preference_context)
                
                if normalized_date is None:
                    if i < MAX_RETRIES: 
                        if not self._handle_nlp_failure(): continue
                        else: return False
                elif normalized_date == "no_dates_for_specialty": 
                     self._speak('no_dates_for_specialty_in_city', specialty=specialty, city=city_display_name); return False 
                elif normalized_date: 
                    chosen_date = normalized_date; break 
                
                if i < MAX_RETRIES: self._speak('date_unavailable_retry_natural', available_dates=", ".join(available_dates)) 
                else: self._speak('date_selection_failed'); return False
            if not chosen_date: return False
        
        confirmation_message_key = 'confirm_appointment_details_ASAP' if date_preference_context or locals().get('current_date_preference_context', "") else 'confirm_appointment_details'
        pref_for_dialogue = date_preference_context or locals().get('current_date_preference_context', "")
        
        confirm_input_stt = self._get_user_input_voice(confirmation_message_key,
                                             specialty=specialty, city=city_display_name, date=chosen_date, 
                                             context_preference=pref_for_dialogue or "",
                                             phrases_for_stt=["da", "nu", "confirm", "anulez", "corect", "greșit", "sigur", "desigur"])
        if not confirm_input_stt: return False

        confirmation = self.nlp.parse_confirmation(confirm_input_stt)
        if confirmation is None: return self._handle_nlp_failure()

        if confirmation == 'da':
            if self.kb.book_appointment(self.current_patient_name, city, specialty, chosen_date):
                self._speak('appointment_confirmed', date=chosen_date, specialty=specialty, city=city_display_name) 
                return True
            self._speak('appointment_booking_failed'); return False
        elif confirmation == 'nu': self._speak('appointment_cancelled_by_user') 
        else: self._speak('confirmation_unclear_appointment_not_made') 
        return False


    def _handle_cancellation(self):
        self._speak('start_cancellation_flow')
        patient_appointments = self.kb.get_patient_appointments(self.current_patient_name) 
        if not patient_appointments: 
            self._speak('no_appointments_found_for_name', name=self.current_patient_name.title())
            return False

        if len(patient_appointments) == 1:
            appt = patient_appointments[0]
            self._speak('found_one_appointment', 
                        specialty=appt['specialty'], date=appt['date'], city=DISPLAY_CITIES.get(appt['city'], appt['city'].title()))
        else:
            self._speak('found_appointments_intro_natural', count=len(patient_appointments))
            for idx, appt in enumerate(patient_appointments):
                pos_desc = ""
                if idx == 0: pos_desc = "Prima programare este "
                elif idx == 1: pos_desc = "A doua este "
                self._speak('appointment_summary_natural', 
                            number=idx + 1, position=pos_desc, 
                            specialty=appt['specialty'], date=appt['date'], city=DISPLAY_CITIES.get(appt['city'], appt['city'].title()))

        selected_appt = None
        phrases_for_cancellation_selection = ["prima", "a doua", "ultima", "cea de la", "programarea din"] + \
                                             [s['specialty'].lower() for s in patient_appointments] + \
                                             [d['date'] for d in patient_appointments] + \
                                             [DISPLAY_CITIES.get(c['city'], c['city']).lower() for c in patient_appointments] 
        
        for i in range(MAX_RETRIES + 1):
            selection_input_stt = self._get_user_input_voice('ask_cancel_selection_description', phrases_for_stt=list(set(phrases_for_cancellation_selection))) 
            if not selection_input_stt: 
                if i < MAX_RETRIES: continue
                else: self._speak('cancellation_selection_failed'); return False
            
            selected_appt = self.nlp.identify_appointment_from_description(selection_input_stt, patient_appointments)
            if selected_appt is None: 
                if i < MAX_RETRIES: 
                    if not self._handle_nlp_failure(): continue
                    else: return False
            elif selected_appt: break 
            
            if i < MAX_RETRIES: self._speak('cancellation_description_not_clear_retry')
        
        if not selected_appt: self._speak('cancellation_selection_failed'); return False
        
        self._speak('understood_appointment_for_cancellation', specialty=selected_appt['specialty'], date=selected_appt['date'])
        confirm_input_stt = self._get_user_input_voice('cancelation_confirm_prompt_specific', 
                                                 date=selected_appt['date'], 
                                                 specialty=selected_appt['specialty'],
                                                 phrases_for_stt=["da", "nu", "confirm", "anulez", "corect", "greșit", "sigur", "desigur"])
        if not confirm_input_stt: return False

        confirmation = self.nlp.parse_confirmation(confirm_input_stt)
        if confirmation is None: return self._handle_nlp_failure()

        if confirmation == 'da':
            if self.kb.cancel_appointment(selected_appt): 
                self._speak('cancel_confirmed_specific',date=selected_appt['date'],specialty=selected_appt['specialty'])
                return True
            self._speak('cancellation_failed_system_error'); return False
        elif confirmation == 'nu': self._speak('cancelation_aborted_by_user')
        else: self._speak('confirmation_unclear_cancellation_not_done')
        return False

    def _handle_redirect_or_operator(self):
        self._speak('specialty_not_found_suggest_operator')


# --- Funcția Principală ---
def main():
    if not os.path.exists(KB_PATH): print(f"({datetime.now().strftime('%H:%M:%S')}) EROARE KB: {KB_PATH} negasit."); return
    if not os.path.exists(DIALOGUES_PATH): print(f"({datetime.now().strftime('%H:%M:%S')}) EROARE Dialoguri: {DIALOGUES_PATH} negasit."); return

    kb_manager = KnowledgeBaseManager(KB_PATH)
    nlp_engine = NLPEngine(client, kb_manager) 
    dialog_manager = DialogManager(nlp_engine, kb_manager, DIALOGUES_PATH)
    
    if not dialog_manager.dialogues or len(dialog_manager.dialogues) < 5: 
        print(f"({datetime.now().strftime('%H:%M:%S')}) EROARE: Dialogurile nu s-au încărcat corect sau sunt insuficiente.")
        return 

    dialog_manager.start()

if __name__ == "__main__":
    main()
