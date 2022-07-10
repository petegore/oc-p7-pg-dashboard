from dashboard_functions import *

st.set_page_config(layout="wide")

# Initializing session state
if "client_id" not in st.session_state:
    st.session_state.client_id = ''

if 'action' not in st.session_state:
    st.session_state.action = 'Parcourir les données'

# Creating sidebar
show_sidebar(st.session_state)

# Creating the main content part
st.title(st.session_state.action)

if st.session_state.action == 'Parcourir les données':
    # Content 1 : data browsing
    show_browsing_content()
elif st.session_state.action == 'Comprendre le modèle':
    # Content 2 : model understanding
    show_model_content()
else:
    # Content 3 : score calculation
    show_scoring_content()
