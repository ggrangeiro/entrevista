import speech_recognition as sr
import pyaudio # Necessário como dependência do speech_recognition para microfone
import google.generativeai as genai
import os
from dotenv import load_dotenv

# --- Configurações Iniciais ---

# Carregar variáveis de ambiente (sua chave da API)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("Erro: Chave da API Gemini não encontrada. Defina GEMINI_API_KEY no arquivo .env")
    exit()

# Configurar a API Gemini
try:
    genai.configure(api_key=GEMINI_API_KEY)
    # Escolha o modelo. 'gemini-pro' é um bom modelo para texto.
    # 'gemini-1.5-flash' é mais rápido e barato, mas 'gemini-1.5-pro' é mais capaz.
    model = genai.GenerativeModel('gemini-2.0-flash') # ou 'gemini-pro' ou 'gemini-1.5-pro-latest'
except Exception as e:
    print(f"Erro ao configurar a API Gemini: {e}")
    exit()

# Configurar Reconhecimento de Fala
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Palavras-chave para identificar perguntas (pode ser melhorado com NLP)
question_keywords = [
    "o que", "qual", "quais", "quem", "como", "onde", "quando", "por que",
    "me diga", "explique", "você pode", "será que", "?"
]

def is_question(text):
    text_lower = text.lower()
    if text_lower.endswith("?"):
        return True
    for keyword in question_keywords:
        # Verifica se a frase começa com a palavra-chave ou contém a palavra-chave com espaços ao redor
        if text_lower.startswith(keyword) or f" {keyword} " in text_lower:
            return True
    return False

def get_gemini_answer(question_text):
    """Envia a pergunta para a API Gemini e retorna a resposta."""
    print("Consultando o Gemini...")
    try:
        # Você pode customizar o prompt para dar mais contexto ou instruções ao Gemini
        # Exemplo: prompt = f"Você é um assistente prestativo. Responda à seguinte pergunta: {question_text}"
        prompt = question_text
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Erro ao chamar a API Gemini: {e}")
        if hasattr(e, 'message') and "API key not valid" in e.message:
             print("Verifique se sua GEMINI_API_KEY está correta e válida.")
        return "Desculpe, não consegui obter uma resposta do Gemini no momento."

# --- Loop Principal ---
print("🎙️ Assistente de Conversa Ativado. Diga 'sair' para terminar.")
print("Ajustando para o ruído ambiente... por favor, aguarde.")

with microphone as source:
    recognizer.adjust_for_ambient_noise(source, duration=2) # Ajusta ao ruído ambiente por 2 segundos
    print("✅ Ajuste de ruído concluído. Pode falar!")

    while True:
        try:
            print("\nOuvindo...")
            # Definir um timeout para não ficar bloqueado indefinidamente se não houver fala
            # phrase_time_limit para o tempo máximo que ele vai ouvir continuamente
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

            print("Reconhecendo fala...")
            # Usando Google Web Speech API (requer internet)
            # Para usar em português: language="pt-BR"
            text = recognizer.recognize_google(audio, language="pt-BR")
            print(f"🗣️ Você disse: {text}")

            if "sair" in text.lower() or "parar" in text.lower():
                print("Encerrando assistente...")
                break

            if is_question(text):
                print("🔍 Detectei uma pergunta!")
                answer = get_gemini_answer(text)
                print(f"🤖 Gemini: {answer}")
            else:
                print("💬 (Não pareceu uma pergunta)")

        except sr.WaitTimeoutError:
            # print("Nenhuma fala detectada no tempo esperado. Continue falando quando quiser.")
            pass # Silenciosamente continua escutando
        except sr.UnknownValueError:
            print("Não consegui entender o áudio. Tente falar mais claramente.")
        except sr.RequestError as e:
            print(f"Erro ao solicitar resultados do serviço de reconhecimento de fala; {e}")
        except KeyboardInterrupt:
            print("\nEncerrando script pelo usuário.")
            break
        except Exception as e:
            print(f"Ocorreu um erro inesperado: {e}")
            break