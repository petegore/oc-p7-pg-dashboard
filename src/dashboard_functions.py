import matplotlib.pyplot as plt
import pandas as pd
import pickle
import requests
import seaborn as sns
import streamlit as st

from lime import lime_tabular

# Loading the dataset used for training the data
TRAIN_SET_PATH = './data/train_set.csv'
TEST_SET_PATH = './data/test_set.csv'
MODEL_PATH = './model/lgbm.pkl'
PREDICTIONS_LABELS = ['CREDIT ACCORDE', 'CREDIT REFUSE']
PREDICTIONS_COLORS = ['darkgreen', 'firebrick']
SCORING_API_URI = "https://infinite-dawn-93145.herokuapp.com/"


@st.cache
def get_train_set():
    """
    Load the train set from the CSV and returns it as a pandas DataFrame.
    The train set is the dataframe on which the model has been trained.

    Returns
        :return: (pd.DataFrame) the train set
    """
    return pd.read_csv(TRAIN_SET_PATH, index_col=0)


@st.cache
def get_test_set():
    """
    Load the test set from the CSV and returns it as a pandas DataFrame.
    The test set is the customers not used for the training.

    Returns
        :return: (pd.DataFrame) the test set
    """
    return pd.read_csv(TEST_SET_PATH, index_col=0)


def show_error(text):
    """ Displays an error text

    Args:
        :param text: (string) the error message
    """
    st.header('Une erreur est survenue')
    st.text(text)
    if st.button("Retour"):
        st.session_state.client_id = ''


def show_sidebar(session_state):
    """ Display the dashboard sidebar """
    with st.sidebar:
        st.title("Menu")
        session_state.action = st.selectbox(
            "Je souhaite",
            ["Parcourir les données", "Calculer un score", "Comprendre le modèle"]
        )


def show_browsing_content():
    """ Display the browsing content page """
    data = get_train_set()

    col_left, col_right = st.columns([2, 3])
    with col_left:
        search_id = st.text_input('Recherchez un client par ID (SK_ID_CURR)')

        if search_id != '':
            search_id = int(search_id)
            data_display = data[data['SK_ID_CURR'] == search_id]
        else:
            data_display = data[:200]
        st.dataframe(data_display)

    with col_right:
        selectable_features = [x for x in data.columns if x not in ['level_0', 'TARGET']]
        selected_feature = st.selectbox(
            'Choisissez une feature à consulter',
            [''] + selectable_features
        )

        if selected_feature != '':
            # We use a sample to improve performances
            feature_range = st.slider(
                'Choisissez un intervalle de valeurs',
                min_value=float(data[selected_feature].min()),
                max_value=float(data[selected_feature].max()),
                value=[float(data[selected_feature].min()), float(data[selected_feature].max())]
            )
            min_selected = feature_range[0]
            max_selected = feature_range[1]

            separate_targets = st.checkbox("Séparer les clients acceptés et refusés")

            if separate_targets:
                fig, axes = plt.subplots(figsize=(15, 5), ncols=2)
                sns.histplot(
                    data=data[
                        (data[selected_feature] >= min_selected)
                        & (data[selected_feature] <= max_selected)
                        & (data['TARGET'] == 0)
                        ][selected_feature],
                    color=PREDICTIONS_COLORS[0],
                    ax=axes[0]
                )
                axes[0].set_title("Crédits acceptés", fontsize=15)
                sns.histplot(
                    data=data[
                        (data[selected_feature] >= min_selected)
                        & (data[selected_feature] <= max_selected)
                        & (data['TARGET'] == 1)
                        ][selected_feature],
                    color=PREDICTIONS_COLORS[1],
                    ax=axes[1]
                )
                axes[1].set_title("Crédits acceptés", fontsize=15)
            else:
                fig = plt.figure(figsize=(15, 7))
                sns.histplot(
                    data=data[
                        (data[selected_feature] >= min_selected)
                        & (data[selected_feature] <= max_selected)
                        ][selected_feature]
                )
            st.pyplot(fig)


def show_scoring_content():
    """ Display the scoring page content """
    col_left, col_right = st.columns(2)

    # DataFrames are set in cache using dedicated functions
    train_set = get_train_set()
    test_set = get_test_set()
    train_sample = train_set.sample(frac=0.2)

    selectable_features = [x for x in train_set.columns if x not in ['level_0', 'TARGET']]

    with col_left:
        search_id = st.text_input('Recherchez un client par son identifiant :')

        if search_id != '':
            search_id = int(search_id)
            predictions = predict_client(search_id)
            if predictions['prediction'] is not None:
                # We display the prediction result with the probability
                plot_score(predictions['prediction'], predictions['probability'])

                # We display the local feature importance
                plot_local_feature_importance(search_id)

                # Then we add a multi-select to choose which features to see
                features_to_show = st.multiselect(
                    'Sur quelles données souhaitez-vous situer le client ?',
                    selectable_features
                )

                for feature_name in features_to_show:
                    plot_client_position(
                        train_set=train_set,
                        client_row=test_set[test_set['SK_ID_CURR'] == search_id],
                        feature_name=feature_name
                    )
            else:
                st.text("Impossible de prédire le score pour ce client")
        else:
            st.dataframe(pd.read_csv(TEST_SET_PATH, nrows=100))

    with col_right:
        if search_id != '':
            st.subheader('Comparez deux features')

            feature_1 = st.selectbox(
                'Vous souhaitez tracer...',
                [''] + selectable_features
            )
            feature_2 = st.selectbox(
                'En fonction de...',
                [''] + selectable_features
            )

            if feature_1 != '' and feature_2 != '':
                fig = plt.figure(figsize=(10, 10))
                sns.scatterplot(
                    data=train_sample,
                    x=feature_2,
                    y=feature_1,
                    hue='TARGET'
                )
                plt.scatter(
                    x=test_set[test_set['SK_ID_CURR'] == search_id][feature_2],
                    y=test_set[test_set['SK_ID_CURR'] == search_id][feature_1],
                    color='red'
                )
                st.pyplot(fig)


def show_model_content():
    """ Display the features importances page """
    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        features = get_model_features_importance()
        nb_features = st.slider(
            "Combien de features souhaitez-vous afficher ?", 1, len(features), 30
        )
        features = dict(sorted(features.items(), key=lambda item: item[1], reverse=True))

        fig = plt.figure(figsize=(20, nb_features))
        sns.barplot(
            x=list(features.values())[:nb_features],
            y=list(features.keys())[:nb_features]
        )
        plt.xlabel("Importance de la feature", fontsize=18)
        st.pyplot(fig)


def get_model_features_importance():
    """ Calls the API to get the model global features importances """
    response = requests.get(SCORING_API_URI + '/features')
    features = response.json()
    return features


def predict_client(client_id):
    """ Calls the API to get the prediction and probability of the client

    Args:
        :param client_id: (int) the ID of the client to predict
    """
    response = requests.get(SCORING_API_URI + str(client_id))
    predictions = response.json()
    return predictions


def plot_score(prediction, probability):
    """ Plot the global score bar of the client

    Args:
        :param prediction: (int) whether the client is risky (1) or safe (0)
        :param probability: (float) the probability of the prediction. Should be greater than 50%
    """
    left_bar_w = probability if prediction == 1 else 1 - probability
    right_bar_w = probability if prediction == 0 else 1 - probability

    fig = plt.figure(figsize=(10, 1))

    plt.title(
        PREDICTIONS_LABELS[prediction],
        fontsize=17,
        color=PREDICTIONS_COLORS[prediction],
        pad=20
    )

    plt.barh(
        y=[0, 0],
        width=[left_bar_w, right_bar_w],
        left=[0, left_bar_w],
        height=2,
        color=list(reversed(PREDICTIONS_COLORS))
    )

    if prediction == 0:
        plt.text(
            left_bar_w + right_bar_w / 2 - 0.05,
            0,
            str(100 * round(right_bar_w, 2)) + '%',
            color='white',
            fontsize=20
        )
    else:
        plt.text(
            left_bar_w / 2 - 0.05,
            0,
            str(100 * round(left_bar_w, 2)) + '%',
            color='white',
            fontsize=20
        )

    plt.yticks([0], [''])
    plt.xticks([0], [''])
    plt.ylim(-1, 1)
    plt.xlim(0, 1)

    st.pyplot(fig)


def plot_client_position(train_set, client_row, feature_name):
    """ Plot the position of current customer into the feature distributions for each target

    Args:
        :param train_set: (pd.DataFrame) the train dataset features
        :param client_row: (pd.Series) the client ID
        :param feature_name: (string) the name of the feature to plot inside X
    """
    fig, axes = plt.subplots(figsize=(20, 8), ncols=2)

    axes[0].set_title("{} (crédits acceptés)".format(feature_name, 0), fontsize=16)
    sns.histplot(
        data=train_set[train_set['TARGET'] == 0],
        x=feature_name,
        ax=axes[0],
        bins=20,
        color=PREDICTIONS_COLORS[1]
    )
    axes[0].vlines(
        x=client_row[feature_name],
        ymin=axes[0].get_ylim()[0],
        ymax=axes[0].get_ylim()[1],
        color='blue',
        linestyle='--'
    )
    axes[0].text(
        client_row[feature_name] * 0.97,
        axes[0].get_ylim()[1] * 0.97,
        "Client",
        color='blue'
    )

    axes[1].set_title("{} (crédits refusés)".format(feature_name), fontsize=16)
    sns.histplot(
        data=train_set[train_set['TARGET'] == 1],
        x=feature_name,
        ax=axes[1],
        bins=20,
        color=PREDICTIONS_COLORS[0]
    )
    axes[1].vlines(
        x=client_row[feature_name],
        ymin=axes[1].get_ylim()[0],
        ymax=axes[1].get_ylim()[1],
        color='blue',
        linestyle='--'
    )
    axes[1].text(
        client_row[feature_name] * 0.97,
        axes[1].get_ylim()[1] * 0.97,
        "Client",
        color='blue'
    )

    st.pyplot(fig)


def plot_local_feature_importance(client_id):
    """ Plot the local feature importance figure

    Args:
        :param client_id: (int) the ID of the client in the test dataset
    """
    train_set = get_train_set()
    test_set = get_test_set()
    train_features = [x for x in train_set.columns if x not in [
        'level_0', 'TARGET', 'SK_ID_CURR', 'isof_score'
    ]]
    test_features = [x for x in test_set.columns if x not in [
        'SK_ID_CURR'
    ]]
    model = pickle.load(open(MODEL_PATH, 'rb'))

    explainer = lime_tabular.LimeTabularExplainer(
        train_set[train_features].to_numpy(),
        mode="classification",
        class_names=PREDICTIONS_LABELS,
        feature_names=train_features,
    )

    explanation = explainer.explain_instance(
        test_set[test_set['SK_ID_CURR'] == client_id][test_features].iloc[0],
        model.predict_proba,
        num_features=20
    )

    st.pyplot(explanation.as_pyplot_figure())
