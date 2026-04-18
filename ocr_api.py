import os
import shutil
import sys
import gc
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# =========================================================
# CẤU HÌNH LOAD OCR (CỔNG 8001)
# =========================================================
PRELOAD_OCR_MODELS = True

if PRELOAD_OCR_MODELS:
    print("Đang tải trước mô hình OCR vào RAM...")
    import ocr_service
else:
    print("Chế độ tiết kiệm RAM (Lazy Load): OCR Model sẽ KHÔNG được load cho đến khi bạn quét ảnh.")

app = FastAPI(title="OCR Service API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.options("/api/v1/extract")
async def extract_ocr_options():
    return {"status": "ok"}

@app.post("/api/v1/extract")
async def extract_ocr(file: UploadFile = File(...)):
    temp_img_path = "temp_upload_image.jpg"
    try:
        with open(temp_img_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if 'ocr_service' not in sys.modules:
            print("Đang tải mô hình OCR vào RAM...")
        import ocr_service

        print(f"Bắt đầu xử lý OCR cho ảnh vừa tải lên...")
        ocr_extracted_data = ocr_service.run_end_to_end_pipeline(temp_img_path)
        
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

        if not ocr_extracted_data:
            return {"status": "error", "message": "OCR không tìm thấy bảng chỉ số nào."}
            
        print("Xử lý OCR thành công!")
        return {"status": "success", "ocr_table": ocr_extracted_data}

    except Exception as e:
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
        print(f"Lỗi hệ thống: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/api/v1/free-ram")
async def free_ram():
    if 'ocr_service' in sys.modules:
        del sys.modules['ocr_service']
        gc.collect()
        return {"status": "success", "message": "Đã giải phóng thành công RAM."}
    return {"status": "info", "message": "OCR hiện chưa được load."}