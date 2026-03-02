"""
app.py — Dash-приложение для интерактивной трёхфакторной оценки рефератов.
"""

import json
import math

import dash
from dash import dcc, html, Input, Output, State, callback, ctx, ALL, no_update
import plotly.graph_objects as go
import pandas as pd

import tsm_engine as eng

# ═══════════════════════════════════════════════════════════════════════
# Конфигурация
# ═══════════════════════════════════════════════════════════════════════

# DEFAULT_METRICS_DIR = 'Seq2Seq/data-metrics-480'
DEFAULT_METRICS_DIR = 'LLMs/data-metrics-480'
DEFAULT_DB_PATH = 'data-tables/data-full+LLM.db'
GRID_COLS = 3  # колонок в сетке 2D графиков

MODEL_SHORT = {
    'summary_lingvo': 'Лингво',
    'summary_TextRank': 'TextRank',
    'summary_LexRank': 'LexRank',
    'summary_mt5': 'mT5',
    'summary_mbart': 'mBART',
    'summary_rut5': 'ruT5',
    'summary_t5': 'T5',
    'summary_Summarunner': 'SummaRuNNer',
    'summary_forzer_GigaChat3-10B-A1.8B_latest': 'GigaChat3',
    'summary_qwen2.5_7b': 'Qwen 2.5',
    'summary_qwen3_8b': 'Qwen 3',
    'summary_yandex_YandexGPT-5-Lite-8B-instruct-GGUF_latest': 'YandexGPT-5',
}

ZONE_LABELS = {
    'Хороший': 'Целевая зона',
    'Копирование': '',
    'Неполный': '',
    'Недост. сжатие': '',
    'Избыт. сжатие': '',
    'Низк. лексика': '',
    'Неоднозначный': '',
}

# ═══════════════════════════════════════════════════════════════════════
# Инициализация
# ═══════════════════════════════════════════════════════════════════════

# Загрузить данные при старте
metrics_data = eng.load_metrics(DEFAULT_METRICS_DIR)
available_models = eng.get_available_models(metrics_data)
available_lex = eng.get_available_lexical_metrics(metrics_data)
available_sem = eng.get_available_semantic_metrics(metrics_data)

# ═══════════════════════════════════════════════════════════════════════
# Приложение Dash
# ═══════════════════════════════════════════════════════════════════════

app = dash.Dash(__name__)
app.title = 'TSM — Трёхфакторная оценка рефератов'

# ═══════════════════════════════════════════════════════════════════════
# Layout
# ═══════════════════════════════════════════════════════════════════════

app.layout = html.Div([
    # Хранилища данных
    dcc.Store(id='store-results', data=None),
    dcc.Store(id='store-calibration', data=None),
    dcc.Store(id='store-thresholds', data=None),
    dcc.Store(id='store-selected-doc', data=None),

    # ── Заголовок ──
    html.Div([
        html.H2('Трёхфакторная оценка качества машинных рефератов',
                style={'margin': 0, 'color': '#2c3e50'}),
        html.P('Интерактивный анализ: лексика × семантика × сжатие',
               style={'margin': 0, 'color': '#7f8c8d', 'fontSize': '14px'}),
    ], style={'padding': '15px 20px', 'borderBottom': '2px solid #3498db',
              'background': '#ecf0f1'}),

    # ── Панель настроек ──
    html.Details([
        html.Summary('Настройки', style={'cursor': 'pointer', 'fontWeight': 'bold',
                                          'fontSize': '15px', 'padding': '8px 0'}),
        html.Div([
            # Строка 1: Метрики
            html.Div([
                html.Div([
                    html.Label('Лексическая метрика', style={'fontWeight': 'bold', 'fontSize': '13px'}),
                    dcc.Dropdown(
                        id='dd-lex-metric',
                        options=[{'label': eng.LEXICAL_LABELS.get(m, m), 'value': m} for m in available_lex],
                        value='rouge1',
                        clearable=False,
                        style={'fontSize': '13px'},
                    ),
                ], style={'flex': '1', 'minWidth': '150px'}),

                html.Div([
                    html.Label('Семантическая метрика', style={'fontWeight': 'bold', 'fontSize': '13px'}),
                    dcc.Dropdown(
                        id='dd-sem-metric',
                        options=[{'label': eng.SEMANTIC_LABELS.get(m, m.replace('emb:', '')), 'value': m}
                                 for m in available_sem],
                        value='bleurt',
                        clearable=False,
                        style={'fontSize': '13px'},
                    ),
                ], style={'flex': '1', 'minWidth': '150px'}),

                html.Div([
                    html.Label('ROUGE мера', style={'fontWeight': 'bold', 'fontSize': '13px'}),
                    dcc.RadioItems(
                        id='radio-rouge-measure',
                        options=[
                            {'label': 'Precision', 'value': 'p'},
                            {'label': 'Recall', 'value': 'r'},
                            {'label': 'F-мера', 'value': 'f'},
                        ],
                        value='p',
                        inline=True,
                        style={'fontSize': '13px'},
                    ),
                ], style={'flex': '1', 'minWidth': '200px'}),
            ], style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap', 'marginBottom': '12px'}),

            # Строка 2: Пороги
            html.Div([
                html.Div([
                    html.Label('Метод порогов', style={'fontWeight': 'bold', 'fontSize': '13px'}),
                    dcc.RadioItems(
                        id='radio-threshold-mode',
                        options=[
                            {'label': 'Процентили', 'value': 'reference'},
                            {'label': '± σ', 'value': 'centered'},
                        ],
                        value='reference',
                        inline=True,
                        style={'fontSize': '13px'},
                    ),
                ], style={'flex': '1'}),

                html.Div([
                    html.Label('Нижн. процентиль', style={'fontSize': '12px'}),
                    dcc.Input(id='input-p-low', type='number', value=10, min=1, max=49,
                              style={'width': '60px', 'fontSize': '13px'}),
                ], style={'flex': '0'}),

                html.Div([
                    html.Label('Верхн. процентиль', style={'fontSize': '12px'}),
                    dcc.Input(id='input-p-high', type='number', value=90, min=51, max=99,
                              style={'width': '60px', 'fontSize': '13px'}),
                ], style={'flex': '0'}),

                html.Div([
                    html.Label('τ (для ±σ)', style={'fontSize': '12px'}),
                    dcc.Input(id='input-tau', type='number', value=2.0, min=0.5, max=5.0, step=0.1,
                              style={'width': '60px', 'fontSize': '13px'}),
                ], style={'flex': '0'}),
            ], style={'display': 'flex', 'gap': '20px', 'alignItems': 'end',
                      'flexWrap': 'wrap', 'marginBottom': '12px'}),

            # Строка 3: Коэффициенты Q
            html.Div([
                html.Label('Коэффициенты Q:', style={'fontWeight': 'bold', 'fontSize': '13px'}),
                html.Span(' α(сем)=', style={'fontSize': '12px'}),
                dcc.Input(id='input-alpha', type='number', value=0.45, min=0, max=1, step=0.05,
                          style={'width': '55px', 'fontSize': '13px'}),
                html.Span(' β(лекс)=', style={'fontSize': '12px'}),
                dcc.Input(id='input-beta', type='number', value=0.25, min=0, max=1, step=0.05,
                          style={'width': '55px', 'fontSize': '13px'}),
                html.Span(' γ(согл)=', style={'fontSize': '12px'}),
                dcc.Input(id='input-gamma', type='number', value=0.15, min=0, max=1, step=0.05,
                          style={'width': '55px', 'fontSize': '13px'}),
                html.Span(' δ(сж)=', style={'fontSize': '12px'}),
                dcc.Input(id='input-delta', type='number', value=0.15, min=0, max=1, step=0.05,
                          style={'width': '55px', 'fontSize': '13px'}),
            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '4px',
                      'flexWrap': 'wrap', 'marginBottom': '12px'}),

            # Строка 4: Модели
            html.Div([
                html.Label('Модели СР:', style={'fontWeight': 'bold', 'fontSize': '13px'}),
                dcc.Checklist(
                    id='checklist-models',
                    options=[{'label': MODEL_SHORT.get(m, m), 'value': m} for m in available_models],
                    value=available_models,
                    inline=True,
                    style={'fontSize': '13px'},
                    inputStyle={'marginRight': '4px', 'marginLeft': '10px'},
                ),
            ], style={'marginBottom': '12px'}),

            # Кнопка
            html.Button('Применить', id='btn-apply', n_clicks=0,
                        style={'backgroundColor': '#3498db', 'color': 'white', 'border': 'none',
                               'padding': '8px 24px', 'fontSize': '14px', 'cursor': 'pointer',
                               'borderRadius': '4px'}),
        ], style={'padding': '10px 0'}),
    ], open=True, style={'padding': '0 20px'}),

    # ── Статистика калибровки ──
    html.Div(id='div-calibration-stats',
             style={'padding': '10px 20px', 'background': '#f8f9fa',
                    'borderBottom': '1px solid #ddd', 'fontSize': '13px'}),

    # ── Основная область: графики + инфо-панель ──
    html.Div([
        # Левая часть: графики
        html.Div([
            # 2D графики (сетка)
            html.Div(id='div-2d-graphs',
                     style={'display': 'grid', 'gridTemplateColumns': f'repeat({GRID_COLS}, 1fr)',
                            'gap': '4px'}),

            # Переключатель 2D/3D
            html.Div([
                dcc.Checklist(
                    id='toggle-3d',
                    options=[{'label': ' Показать 3D график (z_lex × z_sem × z_comp)', 'value': 'show'}],
                    value=[],
                    style={'fontSize': '13px', 'padding': '8px 0'},
                ),
            ]),

            # 3D график
            html.Div(id='div-3d-graph', style={'display': 'none'}),
        ], style={'flex': '1', 'minWidth': '0'}),

        # Правая часть: инфо-панель
        html.Div(id='div-info-panel', children=[
            html.Div('Кликните на точку на графике', style={'color': '#999', 'fontStyle': 'italic'}),
        ], style={'width': '400px', 'minWidth': '360px', 'padding': '10px',
                  'borderLeft': '2px solid #ddd', 'overflowY': 'auto',
                  'maxHeight': '80vh', 'fontSize': '13px'}),
    ], style={'display': 'flex', 'padding': '10px 20px', 'gap': '10px'}),

], style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '1600px', 'margin': '0 auto'})


# ═══════════════════════════════════════════════════════════════════════
# Callback 1: Пересчёт пайплайна
# ═══════════════════════════════════════════════════════════════════════

@callback(
    Output('store-results', 'data'),
    Output('store-calibration', 'data'),
    Output('store-thresholds', 'data'),
    Output('div-calibration-stats', 'children'),
    Input('btn-apply', 'n_clicks'),
    State('dd-lex-metric', 'value'),
    State('dd-sem-metric', 'value'),
    State('radio-rouge-measure', 'value'),
    State('radio-threshold-mode', 'value'),
    State('input-p-low', 'value'),
    State('input-p-high', 'value'),
    State('input-tau', 'value'),
    State('input-alpha', 'value'),
    State('input-beta', 'value'),
    State('input-gamma', 'value'),
    State('input-delta', 'value'),
    State('checklist-models', 'value'),
)
def update_pipeline(n_clicks, lex_mode, sem_mode, rouge_measure,
                    threshold_mode, p_low, p_high, tau,
                    alpha, beta, gamma, delta, selected_models):
    if not selected_models:
        return None, None, None, html.Span('Выберите хотя бы одну модель', style={'color': 'red'})

    try:
        df, cal, thr = eng.run_pipeline(
            metrics_data,
            lex_mode=lex_mode, sem_mode=sem_mode, rouge_measure=rouge_measure,
            selected_models=selected_models,
            threshold_mode=threshold_mode,
            percentile_low=p_low or 10, percentile_high=p_high or 90,
            tau=tau or 2.0,
            alpha=alpha or 0.45, beta=beta or 0.25,
            gamma=gamma or 0.15, delta=delta or 0.15,
        )
    except Exception as e:
        return None, None, None, html.Span(f'Ошибка: {e}', style={'color': 'red'})

    # Статистика
    lex_label = eng.LEXICAL_LABELS.get(lex_mode, lex_mode)
    sem_label = eng.SEMANTIC_LABELS.get(sem_mode, sem_mode.replace('emb:', ''))

    stats = html.Div([
        html.Span(f'Метрики: {lex_label} + {sem_label} | ', style={'fontWeight': 'bold'}),
        html.Span(f'μ_lex={cal["mu_lex"]:.4f}  σ_lex={cal["sigma_lex"]:.4f} | '),
        html.Span(f'μ_sem={cal["mu_sem"]:.4f}  σ_sem={cal["sigma_sem"]:.4f} | '),
        html.Span(f'μ_comp={cal.get("mu_comp", 0):.4f}  σ_comp={cal.get("sigma_comp", 0):.4f} | '),
        html.Span(f'Документов: {cal["n_clean"]} (выбросов: {cal["n_outliers"]}) | '),
        html.Span(f'Моделей: {len(selected_models)}'),
    ])

    return df.to_json(orient='records'), cal, thr, stats


# ═══════════════════════════════════════════════════════════════════════
# Callback 2: Отрисовка 2D графиков
# ═══════════════════════════════════════════════════════════════════════

@callback(
    Output('div-2d-graphs', 'children'),
    Input('store-results', 'data'),
    Input('store-thresholds', 'data'),
    Input('store-selected-doc', 'data'),
)
def update_2d_graphs(results_json, thresholds, selected_doc):
    if results_json is None:
        return [html.Div('Нажмите «Применить»', style={'padding': '40px', 'color': '#999',
                                                        'gridColumn': f'1 / {GRID_COLS + 1}',
                                                        'textAlign': 'center'})]

    df = pd.read_json(results_json, orient='records')
    models = sorted(df['model'].unique())

    graphs = []
    for model in models:
        fig = _make_2d_scatter(df, model, thresholds, selected_doc)
        graph = dcc.Graph(
            id={'type': 'graph-2d', 'index': model},
            figure=fig,
            config={'displayModeBar': False},
            style={'height': '350px'},
        )
        graphs.append(graph)

    return graphs


def _make_2d_scatter(df, model, thresholds, selected_doc):
    """Создать 2D scatter для одной модели."""
    sub = df[df['model'] == model]
    model_label = MODEL_SHORT.get(model, model)

    fig = go.Figure()

    # Пороговая зона (прямоугольник)
    if thresholds:
        tll = thresholds.get('tau_lex_lower', -2)
        tlu = thresholds.get('tau_lex_upper', 2)
        tsl = thresholds.get('tau_sem_lower', -2)
        tsu = thresholds.get('tau_sem_upper', 2)

        fig.add_shape(type='rect', x0=tll, x1=tlu, y0=tsl, y1=tsu,
                      fillcolor='rgba(46,204,113,0.08)', line=dict(color='#2ecc71', width=1, dash='dash'))

    # Точки по диагнозам
    for diag in eng.DIAGNOSIS_COLORS:
        mask = sub['diagnosis_type'] == diag
        if mask.sum() == 0:
            continue
        d = sub[mask]
        fig.add_trace(go.Scatter(
            x=d['z_lex'], y=d['z_sem'],
            mode='markers',
            marker=dict(color=eng.DIAGNOSIS_COLORS[diag], size=6, opacity=0.7,
                        line=dict(width=0)),
            name=eng.DIAGNOSIS_LABELS_RU.get(diag, diag),
            customdata=d[['doc_id', 'Q', 'diagnosis_type', 'diagnosis_confidence', 'z_comp']].values,
            hovertemplate=(
                'doc=%{customdata[0]}<br>'
                'z_lex=%{x:.2f}, z_sem=%{y:.2f}<br>'
                'z_comp=%{customdata[4]:.2f}<br>'
                'Q=%{customdata[1]:.3f}<br>'
                '%{customdata[2]}<extra></extra>'
            ),
            showlegend=False,
        ))

    # Выделенный документ
    if selected_doc is not None:
        sel = sub[sub['doc_id'] == selected_doc]
        if len(sel) > 0:
            fig.add_trace(go.Scatter(
                x=sel['z_lex'], y=sel['z_sem'],
                mode='markers',
                marker=dict(color='red', size=14, symbol='x',
                            line=dict(width=2, color='black')),
                showlegend=False,
                hoverinfo='skip',
            ))

    fig.update_layout(
        title=dict(text=model_label, font=dict(size=13)),
        xaxis=dict(title='z_lex', zeroline=True, zerolinewidth=0.5),
        yaxis=dict(title='z_sem', zeroline=True, zerolinewidth=0.5),
        margin=dict(l=40, r=10, t=35, b=35),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=350,
    )
    fig.update_xaxes(gridcolor='#eee', gridwidth=0.5)
    fig.update_yaxes(gridcolor='#eee', gridwidth=0.5)

    return fig


# ═══════════════════════════════════════════════════════════════════════
# Callback 3: Клик по точке → выбор документа
# ═══════════════════════════════════════════════════════════════════════

@callback(
    Output('store-selected-doc', 'data'),
    Input({'type': 'graph-2d', 'index': ALL}, 'clickData'),
    prevent_initial_call=True,
)
def on_click(click_data_list):
    # Найти какой график был кликнут
    for cd in click_data_list:
        if cd is not None:
            point = cd['points'][0]
            if 'customdata' in point:
                return int(point['customdata'][0])
    return no_update


# ═══════════════════════════════════════════════════════════════════════
# Callback 4: Инфо-панель
# ═══════════════════════════════════════════════════════════════════════

@callback(
    Output('div-info-panel', 'children'),
    Input('store-selected-doc', 'data'),
    State('store-results', 'data'),
)
def update_info_panel(selected_doc, results_json):
    if selected_doc is None or results_json is None:
        return html.Div('Кликните на точку на графике', style={'color': '#999', 'fontStyle': 'italic'})

    df = pd.read_json(results_json, orient='records')
    sub = df[df['doc_id'] == selected_doc].sort_values('Q', ascending=False)

    if len(sub) == 0:
        return html.Div(f'Документ {selected_doc} не найден')

    # Загрузить тексты
    first_model = sub.iloc[0]['model']
    texts = eng.load_texts(DEFAULT_DB_PATH, selected_doc, first_model)

    children = [
        html.H4(f'Документ #{selected_doc}', style={'margin': '0 0 10px 0', 'color': '#2c3e50'}),
    ]

    # Таблица моделей
    table_rows = [html.Tr([
        html.Th('Модель', style={'padding': '4px 8px', 'textAlign': 'left'}),
        html.Th('Диагноз', style={'padding': '4px 8px'}),
        html.Th('Q', style={'padding': '4px 8px'}),
        html.Th('z_lex', style={'padding': '4px 8px'}),
        html.Th('z_sem', style={'padding': '4px 8px'}),
        html.Th('z_comp', style={'padding': '4px 8px'}),
    ], style={'background': '#2c3e50', 'color': 'white', 'fontSize': '11px'})]

    for _, row in sub.iterrows():
        diag = row['diagnosis_type']
        color = eng.DIAGNOSIS_COLORS.get(diag, '#ccc')
        table_rows.append(html.Tr([
            html.Td(MODEL_SHORT.get(row['model'], row['model']),
                    style={'padding': '3px 8px', 'fontWeight': 'bold'}),
            html.Td(eng.DIAGNOSIS_LABELS_RU.get(diag, diag),
                    style={'padding': '3px 8px', 'color': color, 'fontWeight': 'bold'}),
            html.Td(f'{row["Q"]:.3f}', style={'padding': '3px 8px', 'textAlign': 'center'}),
            html.Td(f'{row["z_lex"]:.2f}', style={'padding': '3px 8px', 'textAlign': 'center'}),
            html.Td(f'{row["z_sem"]:.2f}', style={'padding': '3px 8px', 'textAlign': 'center'}),
            html.Td(f'{row["z_comp"]:.2f}', style={'padding': '3px 8px', 'textAlign': 'center'}),
        ], style={'borderBottom': f'2px solid {color}', 'fontSize': '12px'}))

    children.append(html.Table(table_rows, style={
        'borderCollapse': 'collapse', 'width': '100%', 'marginBottom': '15px',
    }))

    # Тексты
    if texts.get('target_summary'):
        target = texts['target_summary']
        children.append(html.Div([
            html.Strong('Авторский реферат', style={'color': '#2c3e50'}),
            html.Span(f' ({len(target)} симв.)', style={'color': '#999', 'fontSize': '11px'}),
            html.Div(target[:500] + ('...' if len(target) > 500 else ''),
                     style={'padding': '6px', 'background': '#f0f0f0', 'borderRadius': '4px',
                            'fontSize': '12px', 'marginTop': '4px', 'whiteSpace': 'pre-wrap'}),
        ], style={'marginBottom': '10px'}))

    # Рефераты моделей (показать лучший и худший)
    if len(sub) > 0:
        best = sub.iloc[0]
        worst = sub.iloc[-1]

        for label, row in [('Лучший', best), ('Худший', worst)]:
            model_col = row['model']
            model_texts = eng.load_texts(DEFAULT_DB_PATH, selected_doc, model_col)
            model_text = model_texts.get('model_summary', '')
            if not model_text:
                continue
            diag = row['diagnosis_type']
            color = eng.DIAGNOSIS_COLORS.get(diag, '#ccc')
            children.append(html.Div([
                html.Strong(f'{label}: {MODEL_SHORT.get(model_col, model_col)}',
                           style={'color': color}),
                html.Span(f' — {eng.DIAGNOSIS_LABELS_RU.get(diag, diag)} (Q={row["Q"]:.3f})',
                          style={'fontSize': '11px', 'color': '#666'}),
                html.Div(model_text[:1000] + ('...' if len(model_text) > 1000 else ''),
                         style={'padding': '6px', 'background': f'{color}11',
                                'borderLeft': f'3px solid {color}', 'borderRadius': '4px',
                                'fontSize': '12px', 'marginTop': '4px', 'whiteSpace': 'pre-wrap'}),
            ], style={'marginBottom': '10px'}))

    return children


# ═══════════════════════════════════════════════════════════════════════
# Callback 5: 3D график
# ═══════════════════════════════════════════════════════════════════════

@callback(
    Output('div-3d-graph', 'children'),
    Output('div-3d-graph', 'style'),
    Input('toggle-3d', 'value'),
    Input('store-results', 'data'),
    Input('store-thresholds', 'data'),
    Input('store-selected-doc', 'data'),
)
def update_3d_graph(toggle_value, results_json, thresholds, selected_doc):
    if 'show' not in (toggle_value or []) or results_json is None:
        return [], {'display': 'none'}

    df = pd.read_json(results_json, orient='records')

    fig = go.Figure()

    # Модели — разные маркеры
    # Scatter3d поддерживает только: circle, circle-open, cross, diamond, diamond-open, square, square-open, x
    symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'circle-open', 'square-open',
               'diamond-open']
    models = sorted(df['model'].unique())

    for i, model in enumerate(models):
        sub = df[df['model'] == model]
        model_label = MODEL_SHORT.get(model, model)

        fig.add_trace(go.Scatter3d(
            x=sub['z_lex'], y=sub['z_sem'], z=sub['z_comp'],
            mode='markers',
            marker=dict(
                color=[eng.DIAGNOSIS_COLORS.get(d, '#ccc') for d in sub['diagnosis_type']],
                size=4, opacity=0.6,
                symbol=symbols[i % len(symbols)],
            ),
            name=model_label,
            customdata=sub[['doc_id', 'Q', 'diagnosis_type']].values,
            hovertemplate=(
                f'{model_label}<br>'
                'doc=%{customdata[0]}<br>'
                'z_lex=%{x:.2f}, z_sem=%{y:.2f}, z_comp=%{z:.2f}<br>'
                'Q=%{customdata[1]:.3f}<br>'
                '%{customdata[2]}<extra></extra>'
            ),
        ))

    # Пороговый параллелепипед (wireframe)
    if thresholds:
        tll = thresholds.get('tau_lex_lower', -2)
        tlu = thresholds.get('tau_lex_upper', 2)
        tsl = thresholds.get('tau_sem_lower', -2)
        tsu = thresholds.get('tau_sem_upper', 2)
        tcl = thresholds.get('tau_comp_lower', -2)
        tcu = thresholds.get('tau_comp_upper', 2)

        # 12 рёбер параллелепипеда
        edges_x, edges_y, edges_z = [], [], []
        for x0, x1 in [(tll, tlu)]:
            for y in [tsl, tsu]:
                for z in [tcl, tcu]:
                    edges_x += [x0, x1, None]
                    edges_y += [y, y, None]
                    edges_z += [z, z, None]
        for y0, y1 in [(tsl, tsu)]:
            for x in [tll, tlu]:
                for z in [tcl, tcu]:
                    edges_x += [x, x, None]
                    edges_y += [y0, y1, None]
                    edges_z += [z, z, None]
        for z0, z1 in [(tcl, tcu)]:
            for x in [tll, tlu]:
                for y in [tsl, tsu]:
                    edges_x += [x, x, None]
                    edges_y += [y, y, None]
                    edges_z += [z0, z1, None]

        fig.add_trace(go.Scatter3d(
            x=edges_x, y=edges_y, z=edges_z,
            mode='lines', line=dict(color='#2ecc71', width=2, dash='dash'),
            name='Целевая зона', showlegend=True,
        ))

    # Выделенный документ
    if selected_doc is not None:
        sel = df[df['doc_id'] == selected_doc]
        if len(sel) > 0:
            fig.add_trace(go.Scatter3d(
                x=sel['z_lex'], y=sel['z_sem'], z=sel['z_comp'],
                mode='markers',
                marker=dict(color='red', size=8, symbol='x',
                            line=dict(width=2, color='black')),
                name=f'doc #{selected_doc}', showlegend=True,
            ))

    fig.update_layout(
        scene=dict(
            xaxis_title='z_lex', yaxis_title='z_sem', zaxis_title='z_comp',
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=550,
        legend=dict(font=dict(size=11)),
    )

    return [dcc.Graph(figure=fig, style={'height': '550px'})], {'display': 'block', 'margin-bottom': '15px', 'border': '1px solid black'}


# ═══════════════════════════════════════════════════════════════════════
# Автозапуск при старте
# ═══════════════════════════════════════════════════════════════════════

# Выполнить пайплайн при первой загрузке
app.clientside_callback(
    """
    function(n) {
        // Нажать кнопку "Применить" при загрузке страницы
        setTimeout(function() {
            var btn = document.getElementById('btn-apply');
            if (btn) btn.click();
        }, 500);
        return '';
    }
    """,
    Output('btn-apply', 'title'),
    Input('btn-apply', 'id'),
)


# ═══════════════════════════════════════════════════════════════════════
# Запуск
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
