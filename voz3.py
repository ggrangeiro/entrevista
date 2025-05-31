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

# Constantes para controle de concisão
MAX_RESPONSE_SENTENCES = 2
MAX_RESPONSE_BULLETS = 3

question_keywords = [
    "o que", "qual", "quais", "quem", "como", "onde", "quando", "por que", "porquê",
    "me diga", "explique", "você pode", "será que", "poderia", "sabe me dizer"
]

# --- SEÇÃO DE PERSONALIZAÇÃO: PREENCHA COM SUAS INFORMAÇÕES ---
MINHAS_INFORMACOES = (
    "PERFIL DO CANDIDATO (USE ESTAS INFORMAÇÕES PARA PERSONALIZAR AS RESPOSTAS):\n"
    "- Nome Referência: Guka (não mencione o nome na resposta, a menos que a pergunta seja 'qual seu nome')\n"
    "- Cargo Almejado: [Ex: Engenheiro de Software Sênior, Cientista de Dados Pleno, etc.]\n"
    "- Anos de Experiência Total: [Ex: 7 anos]\n"
    "- Experiência Principal: [Ex: Desenvolvimento backend com Python e Django, criação de APIs REST, modelagem de dados SQL, implantação em nuvem AWS.]\n"
    "- Habilidades Técnicas Chave: [Ex: Python, Django, Flask, FastAPI, PostgreSQL, MySQL, Docker, Kubernetes, AWS (EC2, S3, Lambda, RDS), Git, CI/CD, Testes Automatizados.]\n"
    "- Habilidades Interpessoais (Soft Skills): [Ex: Liderança técnica, comunicação eficaz, resolução colaborativa de problemas, mentoria de desenvolvedores juniores, proatividade, aprendizado contínuo.]\n"
    "- Conquista Relevante 1 (Exemplo STAR): Situação: Na Empresa Alfa, o sistema de processamento de pedidos era lento e sujeito a erros. Tarefa: Fui encarregado de otimizar o sistema. Ação: Redesenhei a arquitetura do módulo de pedidos, implementei caching e otimizei queries SQL. Resultado: Redução de 80% no tempo de processamento e 95% nos erros, melhorando a satisfação do cliente.\n"
    "- Conquista Relevante 2 (Opcional): [Descreva outra conquista importante, se possível no formato STAR resumido]\n"
    "- Formação Acadêmica: [Ex: Bacharel em Ciência da Computação - Universidade X; Pós-graduação em Engenharia de Software - Instituição Y]\n"
    "- Certificações (se houver): [Ex: AWS Certified Solutions Architect, Certificação Python XPTO]\n"
    "- Objetivo Profissional: [Ex: Busco uma posição desafiadora onde possa aplicar minha experiência em [sua área principal] para desenvolver soluções inovadoras, colaborar com equipes talentosas e continuar meu desenvolvimento profissional em uma empresa com cultura de crescimento como a [Nome da Empresa Entrevistadora, se souber].]\n"
    # Adicione mais pontos que julgar relevantes para suas entrevistas
)

# --- SEÇÃO DE PERSONALIZAÇÃO DA ENTREVISTA: PREENCHA ANTES DE CADA ENTREVISTA ---
# Deixe em branco se não tiver as informações, o código tentará lidar com isso.
INFORMACOES_EMPRESA_VAGA = (
    "SOBRE A EMPRESA E VAGA ATUAL (use para contextualizar respostas como 'por que esta empresa'):\n"
    "- Nome da Empresa Entrevistadora: [Preencher com o nome da empresa]\n"
    "- Setor da Empresa: [Preencher com o setor, ex: Tecnologia, Finanças, Saúde]\n"
    "- Cultura da Empresa (se souber): [Ex: Foco em inovação, colaboração, desenvolvimento de pessoas]\n"
    "- Nome da Vaga: [Preencher com o nome da vaga]\n"
    "- Principais Requisitos da Vaga: [Ex: Experiência com microserviços, liderança de projetos, conhecimento em X, Y, Z]\n"
    "- Desafios da Posição (se souber): [Ex: Ajudar na modernização do legado, escalar a plataforma]\n"
    # Adicione mais pontos sobre a empresa/vaga que podem ser úteis
)
# --- FIM DAS SEÇÕES DE PERSONALIZAÇÃO ---


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
    global chat_model, chat_session, MINHAS_INFORMACOES, INFORMACOES_EMPRESA_VAGA
    try:
        genai.configure(api_key=GEMINI_API_KEY)

        generation_config = genai.types.GenerationConfig(
            temperature=0.35 # Baixa temperatura para respostas mais focadas
        )

        chat_model = genai.GenerativeModel(
            'gemini-1.5-flash-latest', # Rápido e bom para tarefas direcionadas
            generation_config=generation_config
        )

        contexto_candidato = MINHAS_INFORMACOES
        
        empresa_preenchida = False
        for linha in INFORMACOES_EMPRESA_VAGA.splitlines():
            if "[Preencher" not in linha and ":" in linha and len(linha.split(":")[1].strip()) > 0:
                empresa_preenchida = True
                break
        
        contexto_completo = contexto_candidato
        if empresa_preenchida:
            contexto_completo += "\n\n" + INFORMACOES_EMPRESA_VAGA
            print("INFO: Contexto da empresa/vaga incluído.")
        else:
            print("INFO: Contexto da empresa/vaga parece não preenchido, usando apenas informações do candidato.")

        system_instruction = (
            "Você é um coach de entrevistas de elite para o CANDIDATO descrito no contexto. "
            "Sua ÚNICA tarefa é gerar respostas EXTREMAMENTE CURTAS, CONCISAS e DIRETAS para perguntas de entrevista, "
            "baseadas EXCLUSIVAMENTE nas informações do CANDIDATO e da EMPRESA/VAGA fornecidas. "
            f"Use no máximo {MAX_RESPONSE_SENTENCES} frases muito curtas OU, se apropriado, {MAX_RESPONSE_BULLETS} marcadores (bullet points) com poucas palavras cada (use '-' ou '*' para marcadores). "
            "EVITE introduções, conclusões, opiniões pessoais ou qualquer texto desnecessário. Seja factual e direto ao ponto essencial. "
            "Para perguntas comportamentais (ex: 'descreva uma situação...'), use a estrutura STAR (Situação, Tarefa, Ação, Resultado) de forma super concisa, referenciando as conquistas do candidato se aplicável. "
            "Priorize clareza, concisão e relevância para a vaga/empresa descrita."
        )
        
        example_user_q = "P: Por que você se interessou por esta vaga?"
        example_model_a = "R:\n- Alinhamento com desafios da vaga usando minha experiência em [habilidade X do candidato].\n- Oportunidade de contribuir para [objetivo da empresa/vaga].\n- Cultura de [valor da empresa] me atrai."

        chat_session = chat_model.start_chat(history=[
            {'role':'user', 'parts': [contexto_completo]},
            {'role':'model', 'parts': ["Entendido. Contexto do candidato e da vaga recebido. Estou pronto para gerar respostas personalizadas, curtas e diretas."]},
            {'role':'user', 'parts': [system_instruction]},
            {'role':'model', 'parts': ["Compreendi as instruções de formato. Fornecerei respostas otimizadas."]},
            {'role':'user', 'parts': [example_user_q]},
            {'role':'model', 'parts': [example_model_a]}
        ])
        print("API Gemini inicializada com contexto do candidato/vaga e instruções para respostas curtas.")
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
def is_question_keywords(text):
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
                    has_interrogative_element = True; break
            if has_interrogative_element: return True
            if sent.root.pos_ in ["AUX", "VERB"] and sent[0].pos_ in ["AUX", "VERB"]: return True
    return False

def post_process_response(text):
    if not text: return ""
    bullets = re.findall(r"^\s*[-*]\s*(.*)", text, re.MULTILINE)
    if bullets and len(bullets) > MAX_RESPONSE_BULLETS:
        print(f"INFO: Resposta original com {len(bullets)} marcadores, truncando para {MAX_RESPONSE_BULLETS}.")
        new_text_parts = []
        count = 0
        non_bullet_intro_lines = []
        for line in text.splitlines():
            if not re.match(r"^\s*[-*]\s*", line) and count == 0:
                non_bullet_intro_lines.append(line)
            elif re.match(r"^\s*[-*]\s*", line):
                if count == 0: 
                    new_text_parts.extend(non_bullet_intro_lines)
                if count < MAX_RESPONSE_BULLETS:
                    new_text_parts.append(line)
                    count += 1
            elif count > 0 : 
                pass
        if not new_text_parts and non_bullet_intro_lines:
             text = "\n".join(non_bullet_intro_lines)
        else:
             text = "\n".join(new_text_parts)
    elif not bullets:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) > MAX_RESPONSE_SENTENCES:
            print(f"INFO: Resposta original com {len(sentences)} sentenças, truncando para {MAX_RESPONSE_SENTENCES}.")
            text = " ".join(sentences[:MAX_RESPONSE_SENTENCES])
            if text and text[-1].isalnum() and sentences[MAX_RESPONSE_SENTENCES-1].strip().endswith(('.', '!', '?')):
                 text += sentences[MAX_RESPONSE_SENTENCES-1].strip()[-1]
    return text.strip()

def get_gemini_answer(question_text):
    global chat_session
    if not chat_session:
        print("Erro: Sessão de chat com Gemini não iniciada.")
        return "Erro interno: sessão de chat não disponível."
    
    print(f"🤖 Consultando o Gemini sobre: \"{question_text}\"")
    try:
        response = chat_session.send_message(question_text)
        raw_answer = response.text.strip()
        processed_answer = post_process_response(raw_answer)
        
        if raw_answer != processed_answer and len(raw_answer) > 0:
            original_display = raw_answer[:100].replace('\n', ' ')
            processed_display = processed_answer[:100].replace('\n', ' ')
            print(f"INFO: Pós-processamento alterou a resposta. Original: '{original_display}...' | Processada: '{processed_display}...'")
        return processed_answer
    except Exception as e:
        error_message = f"Erro ao chamar a API Gemini: {e}"
        print(error_message)
        print(f"Tipo de exceção na API Gemini: {type(e).__name__}") 
        import traceback
        traceback.print_exc()
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
                    if not is_listening_active.is_set(): continue
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
        print(f"Erro de Atributo ao tentar usar o microfone: {e_attr}")
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
                # --- CORREÇÃO APLICADA AQUI ---
                message_text_display = message.parts[0].text.replace('\n', ' ')
                print(f"  [{message.role.capitalize()}]: {message_text_display}")
                # --- FIM DA CORREÇÃO ---
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