# --- HACK for pysqlite3 ---
# MUST be at the top BEFORE any imports that might use sqlite3
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("✅ Sucesso: Trocado sqlite3 por pysqlite3.")
except ImportError:
    print("⚠️ Aviso: pysqlite3 não encontrado, usando sqlite3 do sistema.")
# --- Fim do HACK ---

import os
import tempfile
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import shutil
import datetime

# Langchain Imports
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
# NOVAS IMPORTAÇÕES PARA SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

# --- Configurações e Constantes ---
APP_TITLE = "🤖 Chat RAG Epaminondas v2.4.0" # Nova versão com SelfQueryRetriever
APP_ICON = "🤖"
PAGE_LAYOUT = "wide"

PERSIST_DIRECTORY = Path("db_chroma_epaminondas")
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 400
MODEL_OPTIONS = ['gemini-2.0-flash', 'gemini-1.5-flash-latest', 'gemini-1.0-pro', 'gemini-pro'] 
DEFAULT_MODEL = 'gemini-2.0-flash'
SUPPORTED_FILE_TYPES = {
    "pdf": PyPDFLoader, "docx": UnstructuredWordDocumentLoader, "txt": TextLoader,
    "md": TextLoader, "csv": TextLoader, "log": TextLoader,
}

# ... (o resto das suas configurações e inicializações de API/Embedding permanecem iguais) ...
# --- Configuração da Chave da API ---
try:
    GOOGLE_API_KEY_GENAI = st.secrets.get("GOOGLE_API_KEY") 
except AttributeError: 
    GOOGLE_API_KEY_GENAI = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY_GENAI:
    st.error("Chave da API do Google (GOOGLE_API_KEY) não configurada.")
    st.stop()

try:
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY_GENAI)
except Exception as e:
    st.error(f"Erro ao inicializar GoogleGenerativeAIEmbeddings: {e}.")
    st.stop()

# --- Funções Auxiliares ---
def log_processing_error(filename: str, error_details: str, suggestion: str):
    if 'file_processing_errors' not in st.session_state:
        st.session_state.file_processing_errors = []
    for err in st.session_state.file_processing_errors:
        if err["filename"] == filename and err["error_details"] == error_details: return # Evita duplicatas
    st.session_state.file_processing_errors.append({
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename, "error_details": error_details, "suggestion": suggestion
    })
    st.error(f"Erro ao processar '{filename}': {error_details}. {suggestion}")

def load_documents_from_path(file_path: Path, file_ext: str) -> List[Document]:
    if not file_path.exists(): 
        msg = f"Arquivo não encontrado: {file_path}"; log_processing_error(file_path.name, msg, "Verifique caminho."); return []
    try:
        loader_class = SUPPORTED_FILE_TYPES.get(file_ext)
        if not loader_class: 
            msg = f"Tipo '{file_ext}' não suportado."; log_processing_error(file_path.name, msg, f"Use: {', '.join(SUPPORTED_FILE_TYPES.keys())}."); return []
        loader = loader_class(str(file_path), autodetect_encoding=True) if loader_class == TextLoader else loader_class(str(file_path))
        docs = loader.load()
        # Adiciona o nome do arquivo como metadado 'source' se não existir
        for doc in docs:
            if 'source' not in doc.metadata:
                doc.metadata['source'] = file_path.name 
            elif not doc.metadata['source']: # Se source for None ou vazio
                doc.metadata['source'] = file_path.name
        return docs
    except Exception as e: 
        sug = "Verifique se arquivo não está corrompido ou protegido."; 
        if "password" in str(e).lower(): sug = "PDF protegido por senha? Remova a senha."
        log_processing_error(file_path.name, str(e), sug); return []

def split_documents(docs: List[Document]) -> List[Document]:
    if not docs: return []
    try:
        return RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len).split_documents(docs)
    except Exception as e:
        st.error(f"Erro crítico na divisão de docs: {e}."); return []

def process_uploaded_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> List[Document]:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    if file_ext not in SUPPORTED_FILE_TYPES: 
        msg = f"Tipo '{file_ext}' não suportado."; log_processing_error(uploaded_file.name, msg, f"Use: {', '.join(SUPPORTED_FILE_TYPES.keys())}."); return []
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue()); tmp_file_path = Path(tmp_file.name)
    docs = load_documents_from_path(tmp_file_path, file_ext)
    # Importante: garantir que 'source' no metadado seja o nome original do arquivo
    for doc in docs:
        doc.metadata['source'] = uploaded_file.name 
    chunks = split_documents(docs)
    try: tmp_file_path.unlink()
    except OSError as e: st.warning(f"Não removeu tmp {tmp_file_path}: {e}")
    return chunks

def process_directory(directory_path: Path) -> List[Document]:
    all_chunks: List[Document] = []; files_processed_count = 0; files_error_count = 0
    if not directory_path.is_dir(): st.error(f"'{directory_path}' não é diretório."); return []
    with st.status(f"Processando diretório: {directory_path.name}", expanded=True) as status_main:
        for item in directory_path.rglob("*"):
            if item.is_file():
                ext = item.suffix.lstrip(".").lower()
                if ext in SUPPORTED_FILE_TYPES:
                    st.write(f" Lendo '{item.name}'...")
                    docs = load_documents_from_path(item, ext) 
                    if docs:
                        # Garante que 'source' seja o nome do arquivo relativo ao diretório processado, ou apenas o nome.
                        for doc in docs:
                            try: # Tenta obter o caminho relativo
                                relative_path = item.relative_to(directory_path)
                                doc.metadata['source'] = str(relative_path)
                            except ValueError: # Se não for subcaminho (improvável com rglob de um dir base)
                                doc.metadata['source'] = item.name
                        chunks_from_file = split_documents(docs)
                        if chunks_from_file: all_chunks.extend(chunks_from_file); files_processed_count += 1
                        elif docs : files_error_count +=1
                    else: files_error_count +=1
        if files_processed_count > 0: status_main.update(label=f"Diretório: {files_processed_count} arquivo(s) OK.", state="complete", expanded=False)
        if files_error_count > 0:
            status_label_error = f"{files_error_count} arquivo(s) no diretório falharam."
            if files_processed_count == 0: status_main.update(label=status_label_error, state="error", expanded=True)
            st.warning(status_label_error, icon="⚠️")
        if files_processed_count == 0 and files_error_count == 0:
            status_main.update(label=f"Nenhum arquivo suportado em '{directory_path}'.", state="complete", expanded=False)
    return all_chunks

def get_vector_store(force_new: bool = False) -> Optional[Chroma]:
    if PERSIST_DIRECTORY.exists() and PERSIST_DIRECTORY.is_dir() and not force_new:
        try: return Chroma(persist_directory=str(PERSIST_DIRECTORY), embedding_function=embedding_function)
        except Exception as e: st.warning(f"Não carregou DB: {e}."); return None
    return None

def add_chunks_to_vector_store(chunks: List[Document], vector_store: Optional[Chroma]) -> Optional[Chroma]:
    if not chunks: return vector_store
    try:
        # Garante que todos os chunks tenham 'source' nos metadados
        for chunk in chunks:
            if 'source' not in chunk.metadata or not chunk.metadata['source']:
                chunk.metadata['source'] = 'desconhecido' # Fallback, mas idealmente deve ser definido antes
                # st.warning(f"Chunk sem 'source' encontrado: {chunk.page_content[:50]}...")


        if vector_store: vector_store.add_documents(chunks); st.success(f"{len(chunks)} chunks adicionados.", icon="➕")
        else:
            PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
            vector_store = Chroma.from_documents(chunks, embedding_function, str(PERSIST_DIRECTORY))
            st.success(f"Novo DB criado com {len(chunks)} chunks.", icon="✨")
        vector_store.persist(); return vector_store
    except Exception as e: st.error(f"Erro ao adicionar ao DB: {e}"); return vector_store

def get_rag_chain(llm: ChatGoogleGenerativeAI, retriever: Any) -> Any:
    # Prompt do sistema com o espaço duplo corrigido
    system_prompt = "Você é um assistente de IA especializado em responder perguntas com base no contexto fornecido. Utilize markdown nas respostas. Responda de forma clara e concisa. Analise o contexto. Se a resposta não estiver no contexto, diga que não sabe. Responda em português brasileiro.\n\nContexto:\n{context}"
    prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)
    return create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)

def ask_llm(model_name: str, query: str, vector_store: Chroma) -> Tuple[str, List[Document]]:
    if not vector_store: return "O banco de vetores não está pronto. Adicione documentos.", []
    
    try:
        # LLM principal para geração da resposta final
        llm_answer = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=GOOGLE_API_KEY_GENAI,
            temperature=0.2
        )

        # LLM para o SelfQueryRetriever (pode ser o mesmo ou um mais simples/rápido)
        # Usar temperatura 0 para geração de query mais determinística
        llm_self_query = ChatGoogleGenerativeAI(
            model=model_name, # Para simplicidade, usamos o mesmo. Pode-se otimizar depois.
            google_api_key=GOOGLE_API_KEY_GENAI,
            temperature=0 
        )

        # Descrição dos metadados para o SelfQueryRetriever
        # É crucial que o metadado 'source' esteja presente e correto nos seus chunks
        metadata_field_info = [
            AttributeInfo(
                name="source",
                description="O nome do arquivo ou caminho do documento original. Exemplos: 'manual_produto_X.pdf', 'guia_instalacao.docx'. Use este filtro se a pergunta do usuário mencionar explicitamente um nome de documento ou pedir informações de um documento específico. Se a pergunta for geral e não especificar um documento, não use este filtro.",
                type="string",
            ),
        ]
        document_content_description = "Conteúdo de um trecho (chunk) de um documento técnico ou manual."

        retriever = None
        try:
            st.write("🔄 Tentando usar SelfQueryRetriever...") # Feedback para o dev
            retriever = SelfQueryRetriever.from_llm(
                llm_self_query,
                vector_store, # Sua instância do ChromaDB
                document_content_description,
                metadata_field_info,
                verbose=True, # MUITO útil para debugging. Mostra a query gerada no console.
                # use_original_query=True, # Se True, usa a query original se a tradução falhar.
                                           # Pode ser útil como fallback. Teste o comportamento.
            )
            st.write("✅ SelfQueryRetriever pronto.")
        except Exception as e:
            st.error(f"Erro ao inicializar SelfQueryRetriever: {e}. Usando retriever padrão.")
            # Fallback para o retriever padrão se o SelfQueryRetriever falhar
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={'k': 5} # Número de chunks a recuperar
            )
        
        if retriever is None: # Segurança adicional
             st.error("Falha crítica ao definir o retriever. Usando retriever padrão.")
             retriever = vector_store.as_retriever(search_kwargs={'k': 5})


        rag_response = get_rag_chain(llm_answer, retriever).invoke({"input": query})
        return rag_response.get("answer", "Não foi possível gerar uma resposta com o contexto."), rag_response.get("context", [])
    except Exception as e: 
        st.error(f"Erro ao comunicar com o LLM ou ao processar a cadeia RAG: {e}")
        return "Desculpe, ocorreu um erro ao processar sua pergunta.", []

# Função do navegador de diretórios (do seu código v2.3 - sem o st.dialog por enquanto para manter a base)
def display_directory_browser():
    st.markdown(f"**Navegando em:** `{st.session_state.current_browse_path}`")
    if st.button("📂 Usar este diretório", key="select_current_browsed_dir", use_container_width=True):
        st.session_state.dir_path_input_value = str(st.session_state.current_browse_path)
        st.toast(f"Caminho selecionado: {st.session_state.dir_path_input_value}", icon="📁"); st.rerun()
    if st.session_state.current_browse_path.parent != st.session_state.current_browse_path:
        if st.button("⬆️ Subir um nível (..)", key="go_up_directory_level", use_container_width=True):
            st.session_state.current_browse_path = st.session_state.current_browse_path.parent; st.rerun()
    st.markdown("---"); subdirs = []
    try:
        for item in sorted(st.session_state.current_browse_path.iterdir(), key=lambda p: p.name.lower()):
            if item.is_dir(): subdirs.append(item)
    except PermissionError: st.error(f"Permissão negada: {st.session_state.current_browse_path}"); return
    except FileNotFoundError: st.error(f"Não encontrado: {st.session_state.current_browse_path}"); return
    except Exception as e: st.error(f"Erro ao listar: {st.session_state.current_browse_path}: {e}"); return
    if subdirs:
        st.markdown("Subdiretórios:"); cols = st.columns(3) ; col_idx = 0
        for i, subdir_path in enumerate(subdirs):
            sanitized_name = subdir_path.name.replace(' ', '_').replace('.', '_').replace(':', '_').replace('\\', '_')
            button_key = f"select_subdir_{i}_{sanitized_name}"
            if cols[col_idx].button(f" D {subdir_path.name}", key=button_key, use_container_width=True):
                st.session_state.current_browse_path = subdir_path; st.rerun()
            col_idx = (col_idx + 1) % 3
    else: st.caption("Nenhum subdiretório.")

# --- Inicialização do Estado da Sessão ---
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = get_vector_store()
    if st.session_state.vector_store: st.session_state.db_loaded_on_startup_toast_pending = True
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'processing_log' not in st.session_state: st.session_state.processing_log = []
if 'file_processing_errors' not in st.session_state: st.session_state.file_processing_errors = []
if 'current_browse_path' not in st.session_state: 
    try: st.session_state.current_browse_path = Path.cwd()
    except Exception: st.session_state.current_browse_path = Path(".").resolve()
if 'dir_path_input_value' not in st.session_state: st.session_state.dir_path_input_value = ""
if 'confirm_delete_db_prompt' not in st.session_state: st.session_state.confirm_delete_db_prompt = False
if 'initial_greeting_added' not in st.session_state: st.session_state.initial_greeting_added = False

if st.session_state.vector_store and not st.session_state.chat_history and not st.session_state.get('initial_greeting_added', False):
    greeting_message = "Olá! Me chamo Epaminondas e estou aqui para te ajudar com informações de suporte com base nas documentações fornecidas."
    st.session_state.chat_history.append({ "query": "", "answer": greeting_message, "sources": [] })
    st.session_state.initial_greeting_added = True

# --- Interface Streamlit ---
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=PAGE_LAYOUT)
st.title(APP_TITLE); st.markdown("Interaja com seus documentos usando o poder da IA Generativa do Google.")

with st.sidebar:
    st.header("⚙️ Configurações e Fontes de Dados")
    st.subheader("🤖 Modelo LLM")
    selected_model = st.selectbox("Selecione o modelo Gemini:", MODEL_OPTIONS, index=MODEL_OPTIONS.index(DEFAULT_MODEL))
    st.divider()
    st.subheader("📄 Upload de Arquivo Único")
    uploaded_file = st.file_uploader("Selecione um arquivo:", type=list(SUPPORTED_FILE_TYPES.keys()), key="file_uploader_widget")
    if uploaded_file:
        if st.button(f"Adicionar '{uploaded_file.name}'", use_container_width=True, type="primary"):
            st.session_state.file_processing_errors = [e for e in st.session_state.file_processing_errors if e.get("filename") != uploaded_file.name]
            with st.spinner(f"Processando '{uploaded_file.name}'..."):
                chunks = process_uploaded_file(uploaded_file)
                if chunks: 
                    st.session_state.vector_store = add_chunks_to_vector_store(chunks, st.session_state.vector_store)
                    st.session_state.processing_log.append(f"'{uploaded_file.name}' adicionado."); 
            st.rerun() 
    st.divider()
    st.subheader("📁 Processar Diretório Local")
    def sync_text_input_to_dir_path_state(): st.session_state.dir_path_input_value = st.session_state.dir_path_widget_key
    current_dir_path_for_input = st.session_state.get('dir_path_input_value', "")
    st.text_input("Caminho do diretório:", value=current_dir_path_for_input, key="dir_path_widget_key", 
                   on_change=sync_text_input_to_dir_path_state, placeholder="/caminho/docs",
                   help="Forneça o caminho ou use o navegador.")
    with st.expander("Procurar Diretório no Servidor", expanded=False): display_directory_browser()
    if st.button("Processar Diretório Informado", use_container_width=True, key="process_directory_from_input_button"):
        dir_to_process_str = st.session_state.get('dir_path_input_value', "").strip()
        if dir_to_process_str:
            target_path = Path(dir_to_process_str)
            if target_path.is_dir():
                chunks = process_directory(target_path)
                if chunks: 
                    st.session_state.vector_store = add_chunks_to_vector_store(chunks, st.session_state.vector_store)
                    st.session_state.processing_log.append(f"Dir '{target_path.name}' processado.")
                st.rerun() 
            else: st.error(f"'{dir_to_process_str}' não é diretório."); st.session_state.processing_log.append(f"Inválido: '{dir_to_process_str}'.")
        else: st.warning("Informe um diretório.")
    st.divider() 
    if st.session_state.vector_store:
        if st.button("🗑️ Limpar Base", use_container_width=True, type="secondary", key="initiate_clear_db"):
            st.session_state.confirm_delete_db_prompt = True 
        if st.session_state.get("confirm_delete_db_prompt", False):
            st.warning("⚠️ Atenção! Remove TODOS os dados da base.")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Confirmar Remoção", use_container_width=True, type="primary", key="confirm_delete_action"):
                    try:
                        if PERSIST_DIRECTORY.exists(): shutil.rmtree(PERSIST_DIRECTORY)
                        st.session_state.vector_store = None; st.session_state.chat_history = []
                        st.session_state.processing_log.append("Base limpa.")
                        st.session_state.file_processing_errors = [] 
                        st.toast("Base limpa!", icon="♻️"); 
                        st.session_state.confirm_delete_db_prompt = False
                        st.session_state.initial_greeting_added = False 
                        st.rerun()
                    except Exception as e: st.error(f"Erro ao limpar: {e}"); st.session_state.confirm_delete_db_prompt = False
            with col2:
                if st.button("Cancelar Remoção", use_container_width=True, key="cancel_delete_action"):
                    st.session_state.confirm_delete_db_prompt = False; st.rerun()
    if st.session_state.processing_log:
        st.markdown("---"); st.markdown("<h6>Histórico Processamento (Geral):</h6>", unsafe_allow_html=True)
        st.caption("\n".join(f"- {log}" for log in reversed(st.session_state.processing_log[-5:])))

# --- Área Principal para Chat ---
if st.session_state.get('file_processing_errors'):
    st.subheader("⚠️ Notificações de Erro no Processamento de Arquivos")
    for error_info in st.session_state.file_processing_errors:
        with st.expander(f"Falha: {error_info['filename']} ({error_info['timestamp']})", expanded=True): # Expandido por padrão
            st.markdown(f"**Detalhes:** {error_info['error_details']}")
            st.markdown(f"**Sugestão:** {error_info['suggestion']}")
    if st.button("Limpar Notificações de Erro", key="clear_error_notifications"):
        st.session_state.file_processing_errors = []
        st.rerun()
    st.markdown("---")

if not st.session_state.vector_store:
    st.info("👋 Bem-vindo! Me chamo Epaminondas. Para começar, adicione documentos à base de conhecimento.")
else:
    if hasattr(st.session_state, 'db_loaded_on_startup_toast_pending') and st.session_state.db_loaded_on_startup_toast_pending:
        st.toast("Base pré-existente carregada!", icon="🗄️"); st.session_state.db_loaded_on_startup_toast_pending = False
    st.success(f"Base carregada. Pergunte!", icon="✅")
    if st.session_state.chat_history:
        st.subheader("📜 Histórico do Chat")
        for i, chat_item in enumerate(reversed(st.session_state.chat_history)): 
            show_user_query_bubble = True
            if chat_item["query"] == "" and i == (len(st.session_state.chat_history) - 1) and st.session_state.get('initial_greeting_added', False):
                 show_user_query_bubble = False
            if show_user_query_bubble:
                 with st.chat_message("user", avatar="👤"): st.markdown(chat_item["query"])
            with st.chat_message("assistant", avatar=APP_ICON):
                st.markdown(chat_item["answer"])
                if chat_item.get("sources"):
                    with st.expander("Ver fontes citadas"):
                        for idx, doc in enumerate(chat_item["sources"]):
                            st.markdown(f"**Fonte {idx+1}:** `{doc.metadata.get('source', 'N/A')}` ({doc.metadata.get('page', 'N/P') if 'page' in doc.metadata else ''})")
                            st.caption(f"> {doc.page_content[:200]}...") 
        st.markdown("---")
    user_query = st.chat_input("Digite sua pergunta sobre os documentos:")
    if user_query:
        with st.spinner(f"Consultando {selected_model}..."):
            answer, sources = ask_llm(selected_model, user_query, st.session_state.vector_store)
            st.session_state.chat_history.append({"query": user_query, "answer": answer, "sources": sources})
            st.rerun()

st.sidebar.markdown("---"); st.sidebar.markdown(f"<small>{APP_TITLE}</small>", unsafe_allow_html=True)
if st.session_state.vector_store:
    try: st.sidebar.caption(f"Docs na base: {st.session_state.vector_store._collection.count()}")
    except Exception: st.sidebar.caption("Base ativa.")