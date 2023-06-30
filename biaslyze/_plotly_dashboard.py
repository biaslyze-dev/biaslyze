"""This file contains the new plotting with plotly and dash."""

import dash
import json
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

concepts = ["Gender", "Religion", "Gender"]
num_columns = 10

# Prepare sample data
religion_df = counterfactual_detection_results._get_result_by_concept(concept="religion")
sort_index = religion_df.median().abs().sort_values(ascending=True)
religion_df = religion_df[sort_index.index]

gender_df = counterfactual_detection_results._get_result_by_concept(concept="gender")
sort_index = gender_df.median().abs().sort_values(ascending=True)
gender_df = gender_df[sort_index.index]


# create a dataframe of texts
religion_text_dict ={
    keyword: [f"{sample.orig_keyword}: {sample.text}" for sample in counterfactual_detection_results._get_counterfactual_samples_by_concept("religion") if sample.keyword == keyword]
    for keyword in religion_df.columns
}

gender_text_dict ={
    keyword: [(sample.orig_keyword, sample.text) for sample in counterfactual_detection_results._get_counterfactual_samples_by_concept("gender") if sample.keyword == keyword]
    for keyword in gender_df.columns
}

dataframes = [(gender_df, gender_text_dict), (religion_df, religion_text_dict), (gender_df, gender_text_dict)]


app = dash.Dash(__name__)


def generate_box_plot(data, num_columns):
    df_subset = data[0].iloc[:, -num_columns:]
    fig = go.Figure()
    for column in df_subset.columns:
        hover_text = [f'Value: {value:.3}<br>Original keyword: {data[1][column][i][0]}<br>Text: {data[1][column][i][1]}' for i, value in enumerate(df_subset[column])]
        fig.add_trace(go.Box(x=df_subset[column], orientation='h', name=column, hovertext=hover_text))
    fig.update_layout(showlegend=False)
    return fig

app.layout = html.Div([
    html.Div(id='button-container', children=[
        html.Button(f'{concepts[i]}', id={'type': 'dataframe-button', 'index': i}, n_clicks=0,
                    style={'background-color': '#4CAF50', 'border': 'none', 'color': 'white', 'padding': '10px 24px', 'text-align': 'center', 'text-decoration': 'none', 'display': 'inline-block', 'font-size': '16px', 'margin': '4px 2px', 'cursor': 'pointer'})
        for i in range(len(dataframes))
    ], style={'margin-bottom': '20px'}),
    dcc.Graph(id='box-plot'),
    dcc.Store(id='selected-dataframe', data=0),
    html.Div(id='selected-text')
])

@app.callback(
    Output('box-plot', 'figure'),
    Output('selected-dataframe', 'data'),
    [Input({'type': 'dataframe-button', 'index': i}, 'n_clicks') for i in range(len(dataframes))],
    State('selected-dataframe', 'data'),
    prevent_initial_callback=True
)
def update_box_plot(*button_clicks_and_selected_dataframe):
    button_clicks = button_clicks_and_selected_dataframe[:-1]
    selected_dataframe = button_clicks_and_selected_dataframe[-1]

    if not any(button_clicks):
        # No button click, use the default selected dataframe (first dataframe)
        fig = generate_box_plot(dataframes[selected_dataframe], num_columns)
        return fig, selected_dataframe

    ctx = dash.callback_context
    triggered_button_id = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
    clicked_dataframe_index = triggered_button_id['index']

    selected_dataframe = int(clicked_dataframe_index)
    fig = generate_box_plot(dataframes[selected_dataframe], num_columns)
    return fig, selected_dataframe

@app.callback(
    Output('selected-text', 'children'),
    Input('box-plot', 'clickData')
)
def display_selected_text(click_data):
    if click_data is not None:
        selected_text = click_data['points'][0]['hovertext']
        lines = selected_text.split("<br>")
        return html.Div([
            html.H4('Selected sample:'),
            html.P(lines[0]),
            html.P(lines[1]),
            html.P(lines[2]),
        ], style={'color': 'white', 'background-color': '#4CAF50', 'padding': '10px', 'border-radius': '5px'})
    else:
        return ''

if __name__ == '__main__':
    app.run_server(debug=True)