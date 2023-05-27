SET SERVER_DIRECTORY=%~dp0

cd %SERVER_DIRECTORY%

uvicorn app.main:app --reload %*