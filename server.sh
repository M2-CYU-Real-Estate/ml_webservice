SERVER_DIRECTORY=$( dirname -- "$0"; )

cd $SERVER_DIRECTORY

uvicorn app.main:app --reload $@