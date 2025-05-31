import speech_recognition as sr
# import pyaudio # Dependência, não precisa importar diretamente
import google.generativeai as genai
import os
from dotenv import load_dotenv
import spacy
import threading
import time
import re # Para o pós-processamento

# --- Configurações Globais e Variáveis ---
GEMINI_API_KEY = None
chat_model = None
chat_session = None
nlp = None
recognizer = None
microphone = None
is_listening_active = threading.Event()

MAX_RESPONSE_SENTENCES = 9 # Para o pós-processamento
MAX_RESPONSE_BULLETS = 6  # Para o pós-processamento

question_keywords = [
    "o que", "qual", "quais", "quem", "como", "onde", "quando", "por que", "porquê",
    "me diga", "explique", "você pode", "será que", "poderia", "sabe me dizer"
]

# --- Funções de Inicialização ---
def load_env_and_api_keys():
    global GEMINI_API_KEY
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("Erro Crítico: Chave da API Gemini (GEMINI_API_KEY) não encontrada no arquivo .env.")
        return False
    return True

def initialize_gemini():
    global chat_model, chat_session
    try:
        genai.configure(api_key=GEMINI_API_KEY)

        generation_config = genai.types.GenerationConfig(
            temperature=0.35 # Baixa temperatura para respostas mais focadas e menos verbosas
            # top_p=0.9,
            # top_k=40
        )

        chat_model = genai.GenerativeModel(
            'gemini-1.5-flash-latest', # Ou 'gemini-pro' se precisar de mais capacidade, mas flash é bom para rapidez
            generation_config=generation_config
        )

        system_instruction = (
            "Você é um assistente de entrevistas de emprego altamente especializado em tecnologia. "
            "Sua ÚNICA tarefa é fornecer respostas EXTREMAMENTE CURTAS, SUCINTAS e DIRETAS para perguntas de entrevista. "
            "As respostas devem ser otimizadas para leitura rápida em tempo real durante uma chamada. "
            f"Use no máximo {MAX_RESPONSE_SENTENCES} frases muito curtas OU, se apropriado, no máximo {MAX_RESPONSE_BULLETS} marcadores (bullet points) com poucas palavras cada (use '-' ou '*' para marcadores). "
            "EVITE QUALQUER introdução, conclusão, explicação longa ou qualquer texto desnecessário. Vá direto ao ponto essencial. "
            "Se uma pergunta for complexa, foque no aspecto mais crucial em pouquíssimas palavras. "
            "Priorize clareza e concisão acima de tudo."
        )
        
        # Exemplo opcional para guiar o modelo ainda mais
        example_user_q = "P: Fale sobre seus principais pontos fortes."
        example_model_a = "R:\n- Proatividade e iniciativa.\n- Rápido aprendizado.\n- Excelente comunicação e trabalho em equipe."


        chat_session = chat_model.start_chat(history=[
            {'role':'user', 'parts': [system_instruction]},
            {'role':'model', 'parts': ["Entendido. Fornecerei respostas super concisas e diretas, prontas para leitura rápida."]},
            # Descomente as duas linhas abaixo para adicionar o exemplo ao histórico inicial
            # {'role':'user', 'parts': [example_user_q]},
            # {'role':'model', 'parts': [example_model_a]}
        ])
        print("API Gemini inicializada com instruções para respostas curtas e diretas.")
        return True
    except Exception as e:
        print(f"Erro Crítico: Erro ao configurar a API Gemini: {e}")
        return False

def initialize_spacy():
    global nlp
    try:
        nlp = spacy.load("pt_core_news_sm")
        print("spaCy inicializado.")
    except OSError:
        print("Modelo spaCy 'pt_core_news_sm' não encontrado. Tentando baixar...")
        try:
            os.system("python -m spacy download pt_core_news_sm")
            nlp = spacy.load("pt_core_news_sm")
            print("Modelo spaCy baixado e carregado com sucesso.")
        except Exception as e_download:
            print(f"Falha ao baixar/carregar 'pt_core_news_sm': {e_download}")
            print("Continuando sem spaCy para detecção de perguntas (usando keywords).")
            nlp = None
    return True

def initialize_speech_recognition():
    global recognizer, microphone
    recognizer = sr.Recognizer()
    try:
        microphone = sr.Microphone()
        print(f"Reconhecimento de fala e microfone (dispositivo padrão: {microphone.device_index if hasattr(microphone, 'device_index') else 'desconhecido'}) inicializados.")
        return True
    except Exception as e:
        print(f"Erro Crítico: Não foi possível instanciar o objeto Microfone: {e}")
        return False

# --- Funções de Lógica ---
def is_question_keywords(text): # Fallback se spaCy não carregar
    text_lower = text.lower().strip()
    if not text_lower: return False
    if text_lower.endswith("?"): return True
    for keyword in question_keywords:
        if text_lower.startswith(keyword + " ") or f" {keyword} " in text_lower:
            return True
    return False

def is_question_spacy(text):
    if not nlp: return is_question_keywords(text)
    doc = nlp(text.strip())
    if not doc or not doc.text.strip(): return False
    if doc.text.strip().endswith("?"): return True
    if doc and doc.has_annotation("SENT_START"):
        for sent in doc.sents:
            if not sent or not sent.root: continue
            has_interrogative_element = False
            for token in sent:
                if token.tag_ == "PRON_INT" or \
                   (token.pos_ == "PRON" and token.morph.get("PronType") == ["Int"]) or \
                   (token.pos_ == "ADV" and token.morph.get("AdvType") == ["Int"]) or \
                   token.lower_ in ["quem", "quê", "que", "qual", "quais", "onde", "quando", "como", "porquê", "por que", "quanto", "quanta", "cadê"]:
                    has_interrogative_element = True
                    break
            if has_interrogative_element: return True
            if sent.root.pos_ in ["AUX", "VERB"] and sent[0].pos_ in ["AUX", "VERB"]: return True
    return False

def post_process_response(text):
    """
    Pós-processamento leve para tentar garantir concisão.
    Prioriza o prompt engineering, mas isso pode ser uma rede de segurança.
    """
    if not text: return ""

    # 1. Limitar marcadores (bullets)
    bullets = re.findall(r"^\s*[-*]\s*(.*)", text, re.MULTILINE)
    if bullets and len(bullets) > MAX_RESPONSE_BULLETS:
        # Se muitos marcadores, pega apenas os primeiros MAX_RESPONSE_BULLETS
        print(f"INFO: Resposta original com {len(bullets)} marcadores, truncando para {MAX_RESPONSE_BULLETS}.")
        new_text_parts = []
        count = 0
        for line in text.splitlines():
            if re.match(r"^\s*[-*]\s*", line):
                if count < MAX_RESPONSE_BULLETS:
                    new_text_parts.append(line)
                    count += 1
            else: # Mantém linhas que não são marcadores (ex: uma introdução curta antes dos marcadores)
                new_text_parts.append(line)
        text = "\n".join(new_text_parts)
        # Poderia adicionar um "..." se truncado, mas para leitura rápida pode ser melhor não ter.
    elif not bullets: # Se não houver marcadores, tenta limitar por sentenças
        # 2. Limitar por sentenças (separação simples, pode não ser perfeita para todas as línguas/casos)
        # Usar regex para uma melhor separação de sentenças considerando abreviações etc. é mais complexo.
        # Esta é uma abordagem simples:
        sentences = re.split(r'(?<=[.!?])\s+', text) # Separa por ., !, ? seguido de espaço
        if len(sentences) > MAX_RESPONSE_SENTENCES:
            print(f"INFO: Resposta original com {len(sentences)} sentenças, truncando para {MAX_RESPONSE_SENTENCES}.")
            text = " ".join(sentences[:MAX_RESPONSE_SENTENCES])
            # Adiciona a pontuação final da última sentença mantida, se ela foi removida pelo split
            if text and text[-1].isalnum() and sentences[MAX_RESPONSE_SENTENCES-1].strip().endswith(('.', '!', '?')):
                 text += sentences[MAX_RESPONSE_SENTENCES-1].strip()[-1]


    # 3. Remover frases de enchimento comuns no final (opcional, pode ser agressivo)
    # fill_phrases_end = ["Em resumo...", "Para concluir...", "Espero que isso ajude."]
    # for phrase in fill_phrases_end:
    #     if text.endswith(phrase):
    #         text = text[:-len(phrase)].strip()

    return text.strip()


def get_gemini_answer(question_text):
    global chat_session
    if not chat_session:
        print("Erro: Sessão de chat com Gemini não iniciada.")
        return "Erro interno: sessão de chat não disponível."
    
    print(f"🤖 Consultando o Gemini sobre: \"{question_text}\"")
    try:
        # O prompt de sistema já deve estar configurado para respostas curtas.
        # A pergunta do usuário é enviada diretamente.
        response = chat_session.send_message(question_text)
        raw_answer = response.text.strip()
        
        # Aplica o pós-processamento
        processed_answer = post_process_response(raw_answer)
        
        if raw_answer != processed_answer and len(raw_answer) > 0 : # Se houve mudança e não era vazio
            print(f"INFO: Pós-processamento alterou a resposta. Original: '{raw_answer[:100]}...' | Processada: '{processed_answer[:100]}...'")

        return processed_answer
    except Exception as e:
        error_message = f"Erro ao chamar a API Gemini: {e}"
        print(error_message)
        return f"Desculpe, não consegui obter uma resposta do Gemini. ({type(e).__name__})"

# --- Loop de Escuta ---
def listening_loop():
    global recognizer, microphone, is_listening_active

    print("\n🎙️  Ajustando para o ruído ambiente...")
    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=1.5)
            print("✅ Ajuste de ruído concluído. Pode falar! (Diga 'sair' ou 'parar conversa' para terminar)")

            while is_listening_active.is_set():
                try:
                    print("\n👂 Ouvindo...")
                    audio = recognizer.listen(source, timeout=2, phrase_time_limit=10)
                    if not is_listening_active.is_set(): break

                    print("🔄 Reconhecendo fala...")
                    text = recognizer.recognize_google(audio, language="pt-BR")
                    print(f"🗣️ Você disse: {text}")

                    if not is_listening_active.is_set(): break

                    if "sair" in text.lower() or "parar conversa" in text.lower():
                        print("🔌 Encerrando assistente por comando de voz...")
                        is_listening_active.clear()
                        break
                    
                    answer = get_gemini_answer(text)
                    if is_listening_active.is_set():
                        print(f"🤖 Gemini: {answer}")

                except sr.WaitTimeoutError:
                    if not is_listening_active.is_set(): break
                    continue
                except sr.UnknownValueError:
                    if not is_listening_active.is_set(): break
                    print("🔇 Não consegui entender o áudio.")
                except sr.RequestError as e:
                    if not is_listening_active.is_set(): break
                    print(f"⚠️ Erro no serviço de fala: {e}")
                except Exception as e:
                    if not is_listening_active.is_set(): break
                    error_msg = f"💥 Erro inesperado no loop de escuta: {e}"
                    print(error_msg)
                    import traceback
                    traceback.print_exc()
                    if isinstance(e, AssertionError) and "Audio source must be entered" in str(e):
                        print("Erro crítico com a fonte de áudio. Encerrando.")
                        is_listening_active.clear()
                    time.sleep(1)
    except AttributeError as e_attr:
        print(f"Erro de Atributo ao tentar usar o microfone (pode indicar que não foi encontrado ou não inicializado corretamente): {e_attr}")
        is_listening_active.clear()
        return
    except Exception as e:
        print(f"Erro crítico ao acessar ou configurar o microfone: {e}")
        is_listening_active.clear()
        return

    print("Loop de escuta encerrado.")
    print("\n📜 Histórico da conversa com Gemini (final):")
    if chat_session:
        for message in chat_session.history:
            if message.parts:
                print(f"  [{message.role.capitalize()}]: {message.parts[0].text}")
            else:
                print(f"  [{message.role.capitalize()}]: (Mensagem filtrada ou vazia)")

# --- Função principal ---
def main_application_cli():
    global is_listening_active, microphone, recognizer

    if not load_env_and_api_keys(): return
    if not initialize_gemini(): return
    if not initialize_spacy(): pass
    if not initialize_speech_recognition(): return

    is_listening_active.set()

    listener_thread = threading.Thread(target=listening_loop)
    listener_thread.daemon = True
    listener_thread.start()

    try:
        while listener_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n⌨️  Interrupção pelo teclado recebida. Encerrando...")
        is_listening_active.clear()
    
    if listener_thread.is_alive():
        print("Aguardando thread de escuta finalizar...")
        listener_thread.join(timeout=3)
    
    print("Programa principal encerrado.")

# --- Bloco de Execução ---
if __name__ == "__main__":
    main_application_cli()