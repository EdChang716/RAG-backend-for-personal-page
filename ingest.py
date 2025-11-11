import os, re, glob, mimetypes
from dotenv import load_dotenv
from typing import List, Tuple
from supabase import create_client
from openai import OpenAI
from pypdf import PdfReader
from unstructured.partition.auto import partition
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
SB = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE"])
OA = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

ALLOWED = {".md", ".txt", ".html", ".htm", ".pdf"}
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
SMALL_DOC_KEEP_AS_ONE = 1200  # 文字少的檔案不切片，整段放

# 去掉像 "README (1).md" 的尾碼
def normalize_stem(stem: str) -> str:
    return re.sub(r"\s*\(\d+\)$", "", stem).strip()

def pick_title(text: str, filename: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "): return line[2:80].strip()
        if line: return line[:80]
    return os.path.splitext(filename)[0]

def extract_pdf_pages(path: str) -> List[Tuple[int, str]]:
    reader = PdfReader(path)
    out = []
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        # 清理：連字、空白
        txt = raw.replace("-\n", "").replace("\u00AD\n", "")
        txt = re.sub(r"[ \t]+\n", "\n", txt)
        txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
        out.append((i, txt))
    return out

def chunks_by_pages(pages: List[Tuple[int,str]], max_chars=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    out = []
    buf, start_page, end_page = "", None, None
    for pno, text in pages:
        if not text.strip(): 
            continue
        if buf and len(buf) + len(text) > max_chars:
            out.append((start_page, end_page, buf))
            buf = buf[-overlap:]
            start_page = pno
        if start_page is None: start_page = pno
        buf += ("\n\n" + text) if buf else text
        end_page = pno
    if buf:
        out.append((start_page, end_page, buf))
    return out

def embed(text: str):
    r = OA.embeddings.create(model="text-embedding-3-small", input=text)
    return r.data[0].embedding

def upsert_rows(rows: list):
    if rows:
        SB.table("documents").upsert(rows, on_conflict="id").execute()

def main():
    root = "data"
    files = [f for f in glob.glob(os.path.join(root, "**", "*"), recursive=True) if os.path.isfile(f)]
    files = [f for f in files if os.path.splitext(f)[1].lower() in ALLOWED]
    if not files:
        print("No files under ./data"); return

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    batch = []
    for path in files:
        ext = os.path.splitext(path)[1].lower()
        filename = os.path.basename(path)
        stem = normalize_stem(os.path.splitext(filename)[0])
        mime = mimetypes.guess_type(path)[0] or ("application/pdf" if ext==".pdf" else "text/plain")

        # 讀文字
        try:
            if ext == ".pdf":
                pages = extract_pdf_pages(path)
                if not any(t.strip() for _, t in pages):
                    print(f"⚠️  No text found (maybe scanned PDF): {filename}")
                    continue
                chunks = chunks_by_pages(pages)
                # each chunk with pages
                for idx, (ps, pe, ctext) in enumerate(chunks):
                    vec = embed(ctext)
                    row = {
                        "id": f"{stem}:{idx}",
                        "doc_id": stem,
                        "title": pick_title(ctext, filename),
                        "chunk_idx": idx,
                        "content": ctext,
                        "mime": mime,
                        "metadata": {"path": path, "pages": f"{ps}-{pe}", "page_start": ps, "page_end": pe},
                        "embedding": vec
                    }
                    batch.append(row)
                    if len(batch) >= 100: upsert_rows(batch); batch = []
                print(f"✅ {filename} -> {len(chunks)} chunks")
                continue

            # 非 PDF 用 unstructured 解析
            elements = partition(filename=path, strategy="fast")
            text = "\n\n".join([getattr(el, "text", "") for el in elements if getattr(el, "text", "")]).strip()
            if not text:
                print(f"⚠️  Empty text: {filename}")
                continue

            title = pick_title(text, filename)
            doc_chunks = [text] if len(text) < SMALL_DOC_KEEP_AS_ONE else splitter.split_text(text)

            for idx, ctext in enumerate(doc_chunks):
                vec = embed(ctext)
                row = {
                    "id": f"{stem}:{idx}",
                    "doc_id": stem,
                    "title": title if idx == 0 else f"{title} (part {idx+1})",
                    "chunk_idx": idx,
                    "content": ctext,
                    "mime": mime,
                    "metadata": {"path": path},
                    "embedding": vec
                }
                batch.append(row)
                if len(batch) >= 100: upsert_rows(batch); batch = []

            print(f"✅ {filename} -> {len(doc_chunks)} chunks")

        except Exception as e:
            print(f"❌ Failed on {filename}: {e}")

    upsert_rows(batch)
    print("🎉 Done")

if __name__ == "__main__":
    main()