from fastapi import APIRouter, UploadFile, File

router = APIRouter()


@router.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    # TODO: lưu file.read() vào DATA_DIR/raw/<timestamp>_<filename>
    raise NotImplementedError
