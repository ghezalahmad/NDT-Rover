import streamlit as st

print(f"Streamlit version: {st.__version__}")
print(f"Has experimental_rerun: {'experimental_rerun' in dir(st)}")