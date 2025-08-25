# alinhador_gui.py

# =============================================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# =============================================================================
import customtkinter as ctk
from tkinter import filedialog, Menu
from pathlib import Path
import cv2
import numpy as np
import threading
import json
import os
import webbrowser
import datetime
import shutil
import sys

# =============================================================================
# 2. CONFIGURAÇÕES GLOBAIS E ÁREA DE CUSTOMIZAÇÃO DA MARCA
# =============================================================================

# --- ★ ÁREA DE CUSTOMIZAÇÃO ★ ---
APP_NAME = "GridAligner"
APP_VERSION = "2.2" # MUDANÇA: Versão incrementada com o ajuste de layout
NOME_DO_AUTOR = "Lucas Henrique de Oliveira"
# --------------------------------

# --- Nomes de arquivos padrão ---
DEFAULT_TEMPLATE_NAME = "template_cartao.jpg"
ICON_NAME = "icone.ico"

# --- Lógica de Caminhos ---
if getattr(sys, 'frozen', False):
    data_dir = Path(sys._MEIPASS)
    config_dir = Path(sys.executable).parent
else:
    data_dir = Path(__file__).parent
    config_dir = Path(__file__).parent

CONFIG_FILE = config_dir / "config.json"
DEFAULT_TEMPLATE_PATH = data_dir / DEFAULT_TEMPLATE_NAME
ICON_PATH = data_dir / ICON_NAME

# =============================================================================
# 3. NÚCLEO LÓGICO DE ALINHAMENTO DE IMAGEM
# =============================================================================
def align_page_contour_cleaning(img_bgr, tpl_shape):
    # ... (a função de alinhamento continua exatamente a mesma) ...
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: raise RuntimeError("Nenhum contorno encontrado na imagem.")
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    page_contour = None
    image_area = thresh.shape[0] * thresh.shape[1]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > image_area * 0.2:
            page_contour = approx
            break
    if page_contour is None: raise RuntimeError("Não foi possível encontrar um contorno de 4 lados que pareça ser a página.")
    mask = np.zeros_like(thresh)
    cv2.drawContours(mask, [page_contour], -1, 255, thickness=cv2.FILLED)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    final_cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not final_cnts: raise RuntimeError("Nenhum contorno final encontrado após a limpeza.")
    final_page_contour = sorted(final_cnts, key=cv2.contourArea, reverse=True)[0]
    peri = cv2.arcLength(final_page_contour, True)
    corners = cv2.approxPolyDP(final_page_contour, 0.02 * peri, True)
    if len(corners) != 4:
        rect = cv2.minAreaRect(final_page_contour)
        corners = cv2.boxPoints(rect)
    rect = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=2) if corners.ndim > 2 else corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]; rect[2] = corners[np.argmax(s)]
    diff = np.diff(corners.reshape(4,2), axis=1)
    rect[1] = corners[np.argmin(diff)]; rect[3] = corners[np.argmax(diff)]
    src_pts = rect.astype("float32")
    h, w = tpl_shape
    dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC)

# =============================================================================
# 4. CLASSE PRINCIPAL DA APLICAÇÃO GRÁFICA (GUI)
# =============================================================================
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- CONFIGURAÇÃO DA JANELA PRINCIPAL ---
        self.title(APP_NAME)
        self.geometry("700x650")
        
        if ICON_PATH.exists():
            try:
                self.iconbitmap(ICON_PATH)
            except Exception as e:
                print(f"Aviso: Não foi possível carregar o ícone da janela: {e}")

        self.grid_columnconfigure(1, weight=1)
        # O peso da linha do relatório ainda é 1 para que ela estique ao redimensionar
        self.grid_rowconfigure(4, weight=1)

        # --- VARIÁVEIS DE CONTROLE DA INTERFACE ---
        self.template_path = ctk.StringVar()
        self.input_dir = ctk.StringVar(value="entrada")
        self.output_dir = ctk.StringVar(value="saida")
        self.error_dir = ctk.StringVar()

        # --- CRIAÇÃO DO MENU SUPERIOR (BARRA DE TAREFAS) ---
        self.menubar = Menu(self)
        self.config(menu=self.menubar)
        select_menu = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Ações", menu=select_menu)
        select_menu.add_command(label="Selecionar Outro Template", command=self.select_template_path)
        select_menu.add_separator()
        select_menu.add_command(label="Nova Análise (Zerar)", command=self.reset_ui)
        help_menu = Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Ajuda", menu=help_menu)
        help_menu.add_command(label="Passo a passo", command=self.show_help_window)
        help_menu.add_separator()
        help_menu.add_command(label=f"Sobre {APP_NAME}", command=self.show_about_window)

        # --- WIDGETS (COMPONENTES) DA INTERFACE PRINCIPAL ---
        current_row = 0
        path_frame = ctk.CTkFrame(self)
        path_frame.grid(row=current_row, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        path_frame.grid_columnconfigure(1, weight=1)
        current_row += 1
        ctk.CTkLabel(path_frame, text="Pasta de Entrada:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(path_frame, textvariable=self.input_dir).grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(path_frame, text="Selecionar...", command=self.select_input_folder).grid(row=1, column=2, padx=10, pady=5)
        ctk.CTkLabel(path_frame, text="Pasta de Saída:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(path_frame, textvariable=self.output_dir).grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(path_frame, text="Selecionar...", command=self.select_output_folder).grid(row=2, column=2, padx=10, pady=5)
        ctk.CTkLabel(path_frame, text="Pasta de Erro (Opcional):").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(path_frame, textvariable=self.error_dir).grid(row=3, column=1, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(path_frame, text="Selecionar...", command=self.select_error_folder).grid(row=3, column=2, padx=10, pady=5)
        
        self.start_button = ctk.CTkButton(self, text="INICIAR ALINHAMENTO", command=self.start_alignment_thread, height=40)
        self.start_button.grid(row=current_row, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        current_row += 1
        
        self.progress_bar = ctk.CTkProgressBar(self, mode='determinate'); self.progress_bar.set(0)
        self.progress_bar.grid(row=current_row, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        current_row += 1
        
        self.status_label = ctk.CTkLabel(self, text="Aguardando início...")
        self.status_label.grid(row=current_row, column=0, columnspan=3, padx=10, pady=5, sticky="w")
        current_row += 1
        
        # MUDANÇA: Adiciona uma altura inicial fixa para a caixa de relatório.
        # O valor 120 (em pixels) corresponde a aproximadamente 4cm em uma tela padrão.
        self.report_textbox = ctk.CTkTextbox(self, state="disabled", height=120)
        self.report_textbox.grid(row=current_row, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
        self.grid_rowconfigure(current_row, weight=1)

        # Carrega as configurações APÓS todos os widgets terem sido criados.
        self.load_settings()

    # --- FUNÇÕES DE LÓGICA DA INTERFACE ---
    def load_settings(self):
        """Carrega o caminho do template, com lógica de fallback para o padrão."""
        try:
            with open(CONFIG_FILE, "r") as f:
                settings = json.load(f)
                path_from_config = settings.get("template_path", "")
                if Path(path_from_config).exists():
                    self.template_path.set(path_from_config)
                    self.status_label.configure(text=f"Template '{Path(path_from_config).name}' carregado.")
                    return
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        
        if DEFAULT_TEMPLATE_PATH.exists():
            self.template_path.set(str(DEFAULT_TEMPLATE_PATH))
            self.status_label.configure(text=f"Template padrão '{DEFAULT_TEMPLATE_NAME}' carregado.")
            self.save_settings()
        else:
            self.status_label.configure(text=f"Aviso: Template padrão '{DEFAULT_TEMPLATE_NAME}' não encontrado!")
    
    # ... (Restante do código sem alterações)...
    def save_settings(self):
        settings = {"template_path": self.template_path.get()}
        with open(CONFIG_FILE, "w") as f: json.dump(settings, f)
    def reset_ui(self):
        self.progress_bar.set(0); self.load_settings(); self.start_button.configure(state="normal", text="INICIAR ALINHAMENTO"); self.report_textbox.configure(state="normal"); self.report_textbox.delete("1.0", "end"); self.report_textbox.configure(state="disabled"); print("Interface zerada para nova análise.")
    def select_template_path(self):
        path = filedialog.askopenfilename(title="Selecione um arquivo de template", filetypes=[("Imagens", "*.jpg *.jpeg *.png")])
        if path: self.template_path.set(path); self.status_label.configure(text=f"Template temporário '{Path(path).name}' carregado."); self.save_settings()
    def select_input_folder(self):
        path = filedialog.askdirectory(title="Selecione a pasta de entrada"); 
        if path: self.input_dir.set(path)
    def select_output_folder(self):
        path = filedialog.askdirectory(title="Selecione a pasta de saída"); 
        if path: self.output_dir.set(path)
    def select_error_folder(self):
        path = filedialog.askdirectory(title="Selecione a pasta para salvar imagens com erro"); 
        if path: self.error_dir.set(path)
    def start_alignment_thread(self):
        self.start_button.configure(state="disabled", text="Processando..."); self.report_textbox.configure(state="normal"); self.report_textbox.delete("1.0", "end"); self.report_textbox.configure(state="disabled"); threading.Thread(target=self.alignment_worker, daemon=True).start()
    def update_gui(self, progress=None, status=None, report=None):
        if progress is not None: self.progress_bar.set(progress)
        if status is not None: self.status_label.configure(text=status)
        if report is not None: self.report_textbox.configure(state="normal"); self.report_textbox.delete("1.0", "end"); self.report_textbox.insert("1.0", report); self.report_textbox.configure(state="disabled")
    def alignment_worker(self):
        if not self.template_path.get() or not Path(self.template_path.get()).exists(): self.update_gui(status="Erro: Template não selecionado ou não encontrado."); self.start_button.configure(state="normal", text="INICIAR ALINHAMENTO"); return
        input_p = Path(self.input_dir.get()); output_p = Path(self.output_dir.get()); error_p_str = self.error_dir.get(); output_p.mkdir(exist_ok=True)
        try:
            with open(self.template_path.get(), 'rb') as f: tpl_stream = np.frombuffer(f.read(), np.uint8)
            tpl_bgr = cv2.imdecode(tpl_stream, cv2.IMREAD_COLOR); tpl_h, tpl_w = tpl_bgr.shape[:2]
        except Exception as e: self.update_gui(status=f"Erro ao carregar o template: {e}"); self.start_button.configure(state="normal", text="INICIAR ALINHAMENTO"); return
        exts = ("*.png", "*.jpg", "*.jpeg", "*.tiff", "*.tif"); paths = [p for ext in exts for p in input_p.glob(ext)]; failed_files = []; total_files = len(paths)
        for i, img_path in enumerate(paths):
            self.after(0, self.update_gui, None, f"Processando: {img_path.name} ({i+1} de {total_files})", None)
            self.after(0, self.progress_bar.set, (i + 1) / total_files)
            try:
                img_stream = np.fromfile(img_path, np.uint8); bgr = cv2.imdecode(img_stream, cv2.IMREAD_COLOR)
                if bgr is None: raise RuntimeError("Não foi possível decodificar a imagem.")
                aligned = align_page_contour_cleaning(bgr, (tpl_h, tpl_w))
                output_suffix = '.jpg' if img_path.suffix.lower() in ['.tiff', '.tif'] else img_path.suffix
                out_path = output_p / f"{img_path.stem}_alinhado{output_suffix}"
                params = [cv2.IMWRITE_JPEG_QUALITY, 95] if output_suffix == '.jpg' else []
                is_success, buffer = cv2.imencode(output_suffix, aligned, params)
                if is_success:
                    with open(out_path, "wb") as f: f.write(buffer)
                else: raise RuntimeError("Falha ao codificar a imagem para salvamento.")
            except Exception as e:
                failed_files.append({"arquivo": img_path.name, "erro": str(e)})
                if error_p_str and Path(error_p_str).is_dir():
                    try: shutil.copy2(img_path, Path(error_p_str) / img_path.name)
                    except Exception as copy_e: print(f"ERRO DE CÓPIA: {copy_e}")
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"); success_count = total_files - len(failed_files)
        report_lines = [f"RELATÓRIO DE PROCESSAMENTO ({now})", "="*50, f"Total: {total_files}", f"✅ Sucessos: {success_count}", f"❌ Falhas: {len(failed_files)}"]
        if failed_files:
            report_lines.append("\n--- DETALHES DAS FALHAS ---")
            for item in failed_files: report_lines.append(f"Arquivo: {item['arquivo']}\n  Motivo: {item['erro']}\n")
        final_report_text = "\n".join(report_lines)
        self.after(0, self.update_gui, None, f"Concluído! {success_count} de {total_files} com sucesso.", final_report_text)
        self.after(0, self.start_button.configure, None, {"state": "normal", "text": "INICIAR ALINHAMENTO"})
    def show_help_window(self):
        help_win = ctk.CTkToplevel(self); help_win.title("Passo a passo"); help_win.geometry("500x350"); help_win.transient(self)
        textbox = ctk.CTkTextbox(help_win, wrap="word", font=("Arial", 12)); textbox.pack(expand=True, fill="both", padx=10, pady=10)
        help_text = f"COMO UTILIZAR O {APP_NAME}\n\n1. TEMPLATE:\n- O programa carrega o '{DEFAULT_TEMPLATE_NAME}' por padrão.\n- Para usar outro, vá em 'Ações' -> 'Selecionar Outro Template'.\n\n2. PASTAS:\n- Verifique as pastas de 'Entrada', 'Saída' e 'Erro' (opcional).\n\n3. INICIAR:\n- Clique em 'INICIAR ALINHAMENTO'.\n\n4. NOVA ANÁLISE:\n- Para processar um novo lote de imagens, vá em 'Ações' -> 'Nova Análise (Zerar)'. Isso limpa a tela para a próxima execução."
        textbox.insert("1.0", help_text); textbox.configure(state="disabled")
    def show_about_window(self):
        about_win = ctk.CTkToplevel(self); about_win.title(f"Sobre {APP_NAME}"); about_win.geometry("450x300"); about_win.resizable(False, False); about_win.transient(self)
        def open_link(url): webbrowser.open_new(url)
        ctk.CTkLabel(about_win, text=APP_NAME, font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(20, 5))
        ctk.CTkLabel(about_win, text=f"Versão {APP_VERSION}", font=ctk.CTkFont(size=12)).pack(pady=0)
        ctk.CTkLabel(about_win, text=NOME_DO_AUTOR, font=ctk.CTkFont(size=14)).pack(pady=20)
        ctk.CTkLabel(about_win, text="(21) 9 9955-3585", font=ctk.CTkFont(size=12)).pack(pady=5)
        link_color = "#0052cc"
        github_link = ctk.CTkLabel(about_win, text="GitHub: LucasHO94", text_color=link_color, cursor="hand2", font=ctk.CTkFont(underline=True))
        github_link.pack(pady=5); github_link.bind("<Button-1>", lambda e: open_link("https://github.com/LucasHO94"))
        linkedin_link = ctk.CTkLabel(about_win, text="LinkedIn: lucasho94", text_color=link_color, cursor="hand2", font=ctk.CTkFont(underline=True))
        linkedin_link.pack(pady=5); linkedin_link.bind("<Button-1>", lambda e: open_link("https://www.linkedin.com/in/lucasho94/"))

# =============================================================================
# 5. PONTO DE ENTRADA DA APLICAÇÃO
# =============================================================================
if __name__ == "__main__":
    app = App()
    app.mainloop()