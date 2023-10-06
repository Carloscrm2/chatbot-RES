## INSTRUCTIONS
1. Create your local environment and activate it
2. Run pip install -r requirements.txt
3. Add your OPENAI_API_KEY on a .env file
3. Run uvicorn --host 0.0.0.0 main:app on one terminal (REST server)
4. Run -m streamlit run stapp.py on another terminal (UI)
5. You can now ask questions on the UI regarding the TEST_DOC.pdf