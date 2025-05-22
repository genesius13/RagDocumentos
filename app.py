import os
import tempfile
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import shutil # Importado no topo para a função de limpar base

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

# --- Configurações e Constantes ---
APP_TITLE = "🤖 Chat RAG Epaminondas v2.3" # Versão incrementada
APP_ICON = "📚"
PAGE_LAYOUT = "wide"

PERSIST_DIRECTORY = Path("db_chroma_epaminondas")
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 400
MODEL_OPTIONS = ['gemini-1.5-flash-latest', 'gemini-1.0-pro', 'gemini-pro']
DEFAULT_MODEL = 'gemini-1.5-flash-latest'
SUPPORTED_FILE_TYPES = {
    "pdf": PyPDFLoader, "docx": UnstructuredWordDocumentLoader, "txt": TextLoader,
    "md": TextLoader, "csv": TextLoader, "log": TextLoader,
}

# --- Configuração da Chave da API ---
try:
    GOOGLE_API_KEY_GENAI = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
except AttributeError: # Para desenvolvimento local sem st.secrets
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
def load_documents_from_path(file_path: Path, file_ext: str) -> List[Document]:
    if not file_path.exists(): st.error(f"Arquivo não encontrado: {file_path}"); return []
    try:
        loader_class = SUPPORTED_FILE_TYPES.get(file_ext)
        if not loader_class: st.warning(f"Tipo '{file_ext}' não suportado para {file_path.name}."); return []
        loader = loader_class(str(file_path), autodetect_encoding=True) if loader_class == TextLoader else loader_class(str(file_path))
        return loader.load()
    except Exception as e: st.error(f"Erro ao carregar o arquivo {file_path.name}: {e}"); return []

def split_documents(docs: List[Document]) -> List[Document]:
    if not docs: return []
    return RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len).split_documents(docs)

def process_uploaded_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> List[Document]:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    if file_ext not in SUPPORTED_FILE_TYPES: st.warning(f"Tipo de arquivo '{file_ext}' não suportado."); return []
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue()); tmp_file_path = Path(tmp_file.name)
    docs = load_documents_from_path(tmp_file_path, file_ext); chunks = split_documents(docs)
    try: tmp_file_path.unlink()
    except OSError as e: st.warning(f"Não foi possível remover o arquivo temporário {tmp_file_path}: {e}")
    return chunks

def process_directory(directory_path: Path) -> List[Document]:
    all_chunks: List[Document] = []; files_processed_count = 0
    if not directory_path.is_dir(): st.error(f"O caminho '{directory_path}' não é um diretório válido."); return []
    for item in directory_path.rglob("*"):
        if item.is_file():
            ext = item.suffix.lstrip(".").lower()
            if ext in SUPPORTED_FILE_TYPES:
                with st.spinner(f"Processando {item.name}..."):
                    docs = load_documents_from_path(item, ext); chunks = split_documents(docs)
                    all_chunks.extend(chunks)
                    if chunks: files_processed_count +=1
    if files_processed_count == 0 and not all_chunks: st.info(f"Nenhum arquivo suportado encontrado ou nenhum conteúdo extraído de '{directory_path}'.")
    elif files_processed_count > 0 : st.toast(f"{files_processed_count} arquivo(s) processado(s) do diretório.", icon="📄")
    return all_chunks

def get_vector_store(force_new: bool = False) -> Optional[Chroma]:
    if PERSIST_DIRECTORY.exists() and PERSIST_DIRECTORY.is_dir() and not force_new:
        try: return Chroma(persist_directory=str(PERSIST_DIRECTORY), embedding_function=embedding_function)
        except Exception as e: st.warning(f"Não foi possível carregar o banco de vetores: {e}. Novo será criado."); return None
    return None

def add_chunks_to_vector_store(chunks: List[Document], vector_store: Optional[Chroma]) -> Optional[Chroma]:
    if not chunks: st.info("Nenhum chunk para adicionar ao banco de vetores."); return vector_store
    try:
        if vector_store: vector_store.add_documents(chunks); st.success(f"{len(chunks)} chunks adicionados ao banco existente.", icon="➕")
        else:
            PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)
            vector_store = Chroma.from_documents(documents=chunks, embedding=embedding_function, persist_directory=str(PERSIST_DIRECTORY))
            st.success(f"Novo banco de vetores criado com {len(chunks)} chunks.", icon="✨")
        vector_store.persist(); return vector_store
    except Exception as e: st.error(f"Erro ao adicionar chunks ao banco de vetores: {e}"); return vector_store

def get_rag_chain(llm: ChatGoogleGenerativeAI, retriever: Any) -> Any:
    system_prompt = "Você é um assistente de IA especializado em responder perguntas com base no contexto fornecido. Analise o contexto. Se a resposta não estiver no contexto, diga que não sabe. Responda em português brasileiro.\n\nContexto:\n{context}"
    prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)
    return create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)

def ask_llm(model_name: str, query: str, vector_store: Chroma) -> Tuple[str, List[Document]]:
    if not vector_store: return "O banco de vetores não está pronto. Adicione documentos.", []
    try:
        llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=GOOGLE_API_KEY_GENAI, temperature=0.2, convert_system_message_to_human=True)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 5})
        rag_response = get_rag_chain(llm, retriever).invoke({"input": query})
        return rag_response.get("answer", "Não foi possível gerar uma resposta com o contexto."), rag_response.get("context", [])
    except Exception as e: st.error(f"Erro ao comunicar com o LLM ou ao processar a cadeia RAG: {e}"); return "Desculpe, ocorreu um erro ao processar sua pergunta.", []

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
    except PermissionError: st.error(f"Permissão negada para acessar: {st.session_state.current_browse_path}"); return
    except FileNotFoundError: st.error(f"Diretório não encontrado: {st.session_state.current_browse_path}"); return
    except Exception as e: st.error(f"Erro ao listar diretórios em {st.session_state.current_browse_path}: {e}"); return
    if subdirs:
        st.markdown("Subdiretórios:"); cols = st.columns(3) ; col_idx = 0
        for i, subdir_path in enumerate(subdirs):
            if cols[col_idx].button(f" D {subdir_path.name}", key=f"select_subdir_{str(subdir_path.name).replace('/', '_')}_{i}", use_container_width=True):
                st.session_state.current_browse_path = subdir_path; st.rerun()
            col_idx = (col_idx + 1) % 3
    else: st.caption("Nenhum subdiretório encontrado.")

# --- Inicialização do Estado da Sessão ---
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = get_vector_store()
    if st.session_state.vector_store: st.session_state.db_loaded_on_startup_toast_pending = True
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'processing_log' not in st.session_state: st.session_state.processing_log = []
if 'current_browse_path' not in st.session_state:
    try: st.session_state.current_browse_path = Path.cwd()
    except Exception: st.session_state.current_browse_path = Path(".").resolve()
if 'dir_path_input_value' not in st.session_state: st.session_state.dir_path_input_value = ""
if 'confirm_delete_db_prompt' not in st.session_state: st.session_state.confirm_delete_db_prompt = False

# --- Interface Streamlit ---
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=PAGE_LAYOUT)
st.title(APP_TITLE); st.markdown("Interaja com seus documentos usando o poder da IA Generativa do Google.")

with st.sidebar:
    st.header("⚙️ Configurações e Fontes de Dados")
    st.subheader("🤖 Modelo LLM")
    selected_model = st.selectbox("Selecione o modelo Gemini:", MODEL_OPTIONS, index=MODEL_OPTIONS.index(DEFAULT_MODEL))
    st.divider()
    st.subheader("📄 Upload de Arquivo Único")
    uploaded_file = st.file_uploader("Arraste e solte ou selecione um arquivo:", type=list(SUPPORTED_FILE_TYPES.keys()), key="file_uploader_widget")
    if uploaded_file:
        if st.button(f"Adicionar '{uploaded_file.name}' à Base", use_container_width=True, type="primary"):
            with st.spinner(f"Processando '{uploaded_file.name}'..."):
                chunks = process_uploaded_file(uploaded_file)
                if chunks:
                    st.session_state.vector_store = add_chunks_to_vector_store(chunks, st.session_state.vector_store)
                    st.session_state.processing_log.append(f"Arquivo '{uploaded_file.name}' adicionado."); st.rerun()
                else: st.warning(f"Nenhum conteúdo processável encontrado em '{uploaded_file.name}'.")
    
    st.divider()
    st.subheader("📁 Processar Diretório Local")
    def sync_text_input_to_dir_path_state(): st.session_state.dir_path_input_value = st.session_state.dir_path_widget_key
    current_dir_path_for_input = st.session_state.get('dir_path_input_value', "")
    st.text_input("Caminho do diretório:", value=current_dir_path_for_input, key="dir_path_widget_key", 
                   on_change=sync_text_input_to_dir_path_state, placeholder="/caminho/para/seus/documentos",
                   help="Forneça o caminho completo ou use o navegador abaixo.")
    with st.expander("Procurar Diretório no Servidor", expanded=False):
        display_directory_browser()
    if st.button("Processar Diretório Informado", use_container_width=True, key="process_directory_from_input_button"):
        dir_to_process_str = st.session_state.get('dir_path_input_value', "").strip()
        if dir_to_process_str:
            target_path = Path(dir_to_process_str)
            if target_path.is_dir():
                with st.spinner(f"Analisando arquivos em '{target_path}'..."):
                    chunks = process_directory(target_path)
                    if chunks:
                        st.session_state.vector_store = add_chunks_to_vector_store(chunks, st.session_state.vector_store)
                        st.session_state.processing_log.append(f"Diretório '{target_path.name}' processado.")
            else: st.error(f"O caminho '{dir_to_process_str}' não é um diretório válido."); st.session_state.processing_log.append(f"Tentativa de processar inválido: '{dir_to_process_str}'.")
        else: st.warning("Por favor, informe o caminho de um diretório ou selecione usando o navegador.")

    st.divider() 
    # CORREÇÃO DE SINTAXE E LÓGICA DE CONFIRMAÇÃO APLICADA ABAIXO
    if st.session_state.vector_store:
        if st.button("🗑️ Limpar Base de Conhecimento", use_container_width=True, type="secondary", key="initiate_clear_db"):
            st.session_state.confirm_delete_db_prompt = True # Ativa o prompt de confirmação

        if st.session_state.get("confirm_delete_db_prompt", False):
            st.warning("⚠️ Atenção! Esta ação removerá TODOS os dados da base de conhecimento permanentemente e não pode ser desfeita.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Confirmar Remoção Definitiva", use_container_width=True, type="destructive" if hasattr(st.elements.button, "type") else "primary", key="confirm_delete_action"):
                    try:
                        if PERSIST_DIRECTORY.exists():
                            shutil.rmtree(PERSIST_DIRECTORY)
                        st.session_state.vector_store = None
                        st.session_state.chat_history = []
                        st.session_state.processing_log.append("Base de conhecimento limpa.")
                        st.toast("Base de conhecimento limpa com sucesso!", icon="♻️")
                        st.session_state.confirm_delete_db_prompt = False # Reseta o prompt
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erro ao limpar a base de conhecimento: {e}")
                        st.session_state.confirm_delete_db_prompt = False # Reseta o prompt
            with col2:
                if st.button("Cancelar Remoção", use_container_width=True, key="cancel_delete_action"):
                    st.session_state.confirm_delete_db_prompt = False # Reseta o prompt
                    st.rerun()
            
    if st.session_state.processing_log:
        st.markdown("---")
        st.markdown("<h6>Histórico de Processamento:</h6>", unsafe_allow_html=True)
        log_entries_to_display = "\n".join(f"- {log_entry}" for log_entry in reversed(st.session_state.processing_log[-5:]))
        st.caption(log_entries_to_display)

# --- Área Principal para Chat ---
if not st.session_state.vector_store:
    st.info("👋 Bem-vindo! Para começar, adicione documentos à base de conhecimento usando as opções na barra lateral.")
else:
    if hasattr(st.session_state, 'db_loaded_on_startup_toast_pending') and st.session_state.db_loaded_on_startup_toast_pending:
        st.toast("Base de conhecimento pré-existente carregada com sucesso!", icon="🗄️"); st.session_state.db_loaded_on_startup_toast_pending = False
    st.success(f"Base de conhecimento carregada. Pronta para suas perguntas!", icon="✅")
    if st.session_state.chat_history:
        st.subheader("📜 Histórico do Chat")
        for chat_item in reversed(st.session_state.chat_history): # Mostra mais recentes primeiro
            with st.chat_message("user", avatar="👤"): st.markdown(chat_item["query"])
            with st.chat_message("assistant", avatar=APP_ICON):
                st.markdown(chat_item["answer"])
                if chat_item.get("sources"):
                    with st.expander("Ver fontes citadas"):
                        for idx, doc in enumerate(chat_item["sources"]):
                            st.markdown(f"**Fonte {idx+1}:** `{doc.metadata.get('source', 'Desconhecida')}`")
                            st.caption(f"> {doc.page_content[:200]}...") # Mostra um trecho
        st.markdown("---")
    user_query = st.chat_input("Digite sua pergunta sobre os documentos:")
    if user_query:
        with st.spinner(f"Consultando {selected_model}..."):
            answer, sources = ask_llm(selected_model, user_query, st.session_state.vector_store)
            st.session_state.chat_history.append({"query": user_query, "answer": answer, "sources": sources})
            st.rerun()

st.sidebar.markdown("---"); st.sidebar.markdown(f"<small>{APP_TITLE} | Powered by Langchain & Google Gemini</small>", unsafe_allow_html=True)
if st.session_state.vector_store:
    try: st.sidebar.caption(f"Documentos na base: {st.session_state.vector_store._collection.count()}")
    except Exception: st.sidebar.caption("Base de conhecimento ativa.")