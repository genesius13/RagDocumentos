# google_drive/drive_utils.py
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import tempfile
import os
import streamlit as st # Para st.error em caso de falha na construção do serviço

# A constante SCOPES será definida em app.py, pois faz parte da configuração do fluxo OAuth.
# O caminho para o arquivo de credenciais da conta de serviço não é mais relevante aqui.

def build_drive_service_with_user_creds(credentials):
    """
    Constrói um objeto de serviço do Google Drive usando as credenciais OAuth do usuário.
    """
    try:
        service = build('drive', 'v3', credentials=credentials)
        print("Serviço do Google Drive construído com sucesso com as credenciais do usuário.")
        return service
    except Exception as e:
        print(f"Erro ao construir o serviço do Google Drive com as credenciais do usuário: {e}")
        st.error(f"Erro ao construir o serviço do Google Drive: {e}")
        return None

def list_files(service, parent_folder_id='root', mime_type=None, custom_query=None):
    """
    Lista arquivos e/ou pastas no Google Drive do usuário.
    Adiciona supportsAllDrives e includeItemsFromAllDrives para melhor compatibilidade.
    """
    try:
        query_parts = []
        effective_query = ""
        if custom_query:
            effective_query = custom_query
        else:
            if parent_folder_id:
                query_parts.append(f"'{parent_folder_id}' in parents")
            if mime_type:
                query_parts.append(f"mimeType='{mime_type}'")
            query_parts.append("trashed=false")
            effective_query = " and ".join(query_parts) if query_parts else None
        
        results = service.files().list(
            q=effective_query,
            pageSize=100,
            fields="files(id, name, mimeType, webViewLink, iconLink)",
            orderBy="folder, name",
            supportsAllDrives=True,        # Importante para Drives Compartilhados
            includeItemsFromAllDrives=True # Importante para Drives Compartilhados
        ).execute()
        files = results.get('files', [])
        return files
    except Exception as e:
        print(f"Erro CRÍTICO ao listar arquivos no Google Drive (query='{effective_query if 'effective_query' in locals() else 'N/A'}'): {e}")
        st.error(f"Ocorreu um erro ao tentar listar arquivos do seu Google Drive: {e}. Verifique as permissões ou tente novamente.")
        return []


def download_file(service, file_id, file_name):
    """
    Baixa um arquivo do Google Drive do usuário para um arquivo temporário.
    """
    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        progress_bar = st.progress(0, text=f"Baixando '{file_name}'...") # Barra de progresso
        while not done:
            status, done = downloader.next_chunk()
            if status:
                progress = int(status.progress() * 100)
                progress_bar.progress(progress, text=f"Baixando '{file_name}': {progress}%")
        
        progress_bar.empty() # Limpa a barra de progresso
        fh.seek(0)
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file_name}")
        with temp_file:
            temp_file.write(fh.read())
            temp_file_path = temp_file.name
        return temp_file_path
    except Exception as e:
        if 'progress_bar' in locals(): progress_bar.empty()
        print(f"Erro ao baixar o arquivo '{file_name}' (ID: {file_id}): {e}")
        st.error(f"Erro ao baixar o arquivo '{file_name}': {e}")
        return None