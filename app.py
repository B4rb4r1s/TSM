"""
app.py — Dash-приложение для интерактивной трёхфакторной оценки рефератов.

Две вкладки:
1. Анализ моделей — визуализация метрик из БД
2. Ручной анализ — ввод текста + реферата, вычисление метрик в реальном времени
"""

import dash
from dash import dcc, html, Input, Output, State, callback, ctx, ALL, no_update
import plotly.graph_objects as go
import pandas as pd

import tsm_engine as eng
import metrics_compute as mc
import tsm_config as config
from db import Database
import tsm_db

# ═══════════════════════════════════════════════════════════════════════
# Конфигурация
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_SOURCE = '480'          # publications.source в БД
GRID_COLS = 4                   # колонок в сетке 2D графиков

# MODEL_SHORT строится динамически ниже с учётом версий моделей

# ═══════════════════════════════════════════════════════════════════════
# Инициализация
# ═══════════════════════════════════════════════════════════════════════

# Подключение к БД
db = Database()

# Доступные модели (с версиями) и метрики из БД
model_versions_data = tsm_db.get_model_versions_from_db(db, source=DEFAULT_SOURCE)
MODEL_SHORT = {mv['label']: config.make_model_short_label(mv['name'], mv['version'])
               for mv in model_versions_data}
MODEL_SHORT['reference'] = 'АР (эталон)'
available_lex, available_sem = tsm_db.get_available_metric_modes(db, source=DEFAULT_SOURCE)

# ═══════════════════════════════════════════════════════════════════════
# Приложение Dash
# ═══════════════════════════════════════════════════════════════════════

app = dash.Dash(__name__)
app.title = 'TSM — Трёхфакторная оценка рефератов'

# ═══════════════════════════════════════════════════════════════════════
# Построение группированного списка моделей
# ═══════════════════════════════════════════════════════════════════════

def _build_model_groups(model_versions):
    """Сгруппированные чеклисты моделей по типу версии."""
    # Группировка по версии
    version_models = {}
    seen = set()
    ordered = []

    for v in config.VERSION_ORDER:
        mvs = [mv for mv in model_versions if mv['version'] == v]
        if mvs:
            version_models[v] = mvs
            ordered.append(v)
            seen.add(v)

    for mv in model_versions:
        v = mv['version']
        if v not in seen:
            if v not in version_models:
                version_models[v] = []
                ordered.append(v)
                seen.add(v)
            version_models[v].append(mv)

    children = []
    for version in ordered:
        mvs = version_models[version]
        group_label = config.VERSION_GROUP_LABELS.get(version, version)

        options = []
        defaults = []
        for mv in sorted(mvs, key=lambda x: x['name']):
            short = config.make_model_short_label(mv['name'], mv['version'])
            if mv['has_metrics']:
                opt_label = short
            else:
                opt_label = f"{short} [нет метрик]"
            options.append({
                'label': opt_label,
                'value': mv['label'],
                'disabled': not mv['has_metrics'],
            })
            if mv['has_metrics']:
                defaults.append(mv['label'])

        children.append(html.Div([
            html.Span(group_label + ':', style={
                'fontWeight': 'bold', 'fontSize': '12px', 'color': '#555',
                'minWidth': '200px', 'display': 'inline-block',
            }),
            dcc.Checklist(
                id={'type': 'model-check', 'group': version},
                options=options,
                value=defaults,
                inline=True,
                style={'fontSize': '12px', 'display': 'inline'},
                inputStyle={'marginRight': '3px', 'marginLeft': '8px'},
            ),
        ], style={'marginBottom': '4px', 'display': 'flex', 'alignItems': 'center',
                  'flexWrap': 'wrap'}))

    return children


# ═══════════════════════════════════════════════════════════════════════
# Layout — Вкладка 1: Анализ моделей
# ═══════════════════════════════════════════════════════════════════════

tab_models = dcc.Tab(label='Анализ моделей', value='tab-models', children=[
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
                        value=available_lex[0] if available_lex else 'rouge1',
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
                        value=available_sem[0] if available_sem else 'bertscore',
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

                html.Div([
                    html.Label('BERTScore мера', style={'fontWeight': 'bold', 'fontSize': '13px'}),
                    dcc.RadioItems(
                        id='radio-bertscore-measure',
                        options=[
                            {'label': 'Precision', 'value': 'p'},
                            {'label': 'Recall', 'value': 'r'},
                            {'label': 'F-мера', 'value': 'f'},
                        ],
                        value='f',
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

            # Строка 4: Модели (сгруппированные по версиям)
            html.Div([
                html.Label('Модели СР:', style={'fontWeight': 'bold', 'fontSize': '13px',
                                                 'marginBottom': '8px', 'display': 'block'}),
                html.Div(_build_model_groups(model_versions_data)),
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
                    value=['show'],
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
                  'maxHeight': '180vh', 'fontSize': '13px'}),
    ], style={'display': 'flex', 'padding': '10px 20px', 'gap': '10px'}),
])


# ═══════════════════════════════════════════════════════════════════════
# Layout — Вкладка 2: Ручной анализ
# ═══════════════════════════════════════════════════════════════════════

_textarea_style = {
    'width': '100%', 'minHeight': '120px', 'fontSize': '13px',
    'padding': '8px', 'borderRadius': '4px', 'border': '1px solid #ccc',
    'fontFamily': 'Arial, sans-serif', 'resize': 'vertical',
}

tab_manual = dcc.Tab(label='Ручной анализ', value='tab-manual', children=[
    html.Div([
        # ── Ввод текстов ──
        html.Div([
            html.H4('Ввод текстов', style={'margin': '0 0 10px 0', 'color': '#2c3e50'}),
            html.P('Параметры метрик и калибровка берутся из вкладки «Анализ моделей».',
                   style={'fontSize': '12px', 'color': '#7f8c8d', 'margin': '0 0 12px 0'}),

            # Исходный текст
            html.Div([
                html.Label('Исходный текст (ОТ) *',
                           style={'fontWeight': 'bold', 'fontSize': '13px', 'marginBottom': '4px'}),
                dcc.Textarea(
                    id='textarea-source',
                    placeholder='Вставьте исходный текст статьи...',
                    style={**_textarea_style, 'minHeight': '150px'},
                ),
            ], style={'marginBottom': '12px'}),

            # Реферат
            html.Div([
                html.Label('Реферат (СР) *',
                           style={'fontWeight': 'bold', 'fontSize': '13px', 'marginBottom': '4px'}),
                dcc.Textarea(
                    id='textarea-summary',
                    placeholder='Вставьте реферат для оценки...',
                    style=_textarea_style,
                ),
            ], style={'marginBottom': '12px'}),

            # Авторский реферат (опционально)
            html.Div([
                html.Label('Авторский реферат (АР) — опционально',
                           style={'fontWeight': 'bold', 'fontSize': '13px', 'marginBottom': '4px'}),
                html.Span(' (для полного Q-score)',
                          style={'fontSize': '11px', 'color': '#999'}),
                dcc.Textarea(
                    id='textarea-reference',
                    placeholder='Вставьте авторский реферат (если есть)...',
                    style=_textarea_style,
                ),
            ], style={'marginBottom': '16px'}),

            # Кнопка
            html.Button('Вычислить метрики', id='btn-compute-manual', n_clicks=0,
                        style={'backgroundColor': '#27ae60', 'color': 'white', 'border': 'none',
                               'padding': '10px 30px', 'fontSize': '15px', 'cursor': 'pointer',
                               'borderRadius': '4px', 'fontWeight': 'bold'}),
        ], style={'maxWidth': '900px'}),

        # ── Результаты (внутри Loading) ──
        dcc.Loading(
            id='loading-manual',
            type='default',
            color='#27ae60',
            children=[
                html.Div(id='div-manual-results', style={'marginTop': '20px'}),
            ],
        ),

    ], style={'padding': '15px 20px'}),
])


# ═══════════════════════════════════════════════════════════════════════
# Общий Layout
# ═══════════════════════════════════════════════════════════════════════

app.layout = html.Div([
    # Хранилища данных (общие для обеих вкладок)
    dcc.Store(id='store-results', data=None),
    dcc.Store(id='store-calibration', data=None),
    dcc.Store(id='store-thresholds', data=None),
    dcc.Store(id='store-selected-doc', data=None),
    dcc.Store(id='store-manual-metrics', data=None),

    # ── Заголовок ──
    html.Div([
        html.H2('Трёхфакторная оценка качества машинных рефератов',
                style={'margin': 0, 'color': '#2c3e50'}),
        html.P('Интерактивный анализ: лексика × семантика × сжатие',
               style={'margin': 0, 'color': '#7f8c8d', 'fontSize': '14px'}),
    ], style={'padding': '15px 20px', 'borderBottom': '2px solid #3498db',
              'background': '#ecf0f1'}),

    # ── Вкладки ──
    dcc.Tabs(id='main-tabs', value='tab-models', children=[
        tab_models,
        tab_manual,
    ], style={'fontSize': '14px'}),

], style={'fontFamily': 'Arial, sans-serif', 'maxWidth': '95%', 'margin': '0 auto'})


# ═══════════════════════════════════════════════════════════════════════
# Callback 1: Пересчёт пайплайна (Вкладка 1)
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
    State('radio-bertscore-measure', 'value'),
    State('radio-threshold-mode', 'value'),
    State('input-p-low', 'value'),
    State('input-p-high', 'value'),
    State('input-tau', 'value'),
    State('input-alpha', 'value'),
    State('input-beta', 'value'),
    State('input-gamma', 'value'),
    State('input-delta', 'value'),
    State({'type': 'model-check', 'group': ALL}, 'value'),
)
def update_pipeline(n_clicks, lex_mode, sem_mode, rouge_measure, bertscore_measure,
                    threshold_mode, p_low, p_high, tau,
                    alpha, beta, gamma, delta, selected_models_groups):
    # Собрать выбранные модели из всех групп
    selected_models = [m for group in (selected_models_groups or []) for m in (group or [])]
    if not selected_models:
        return None, None, None, html.Span('Выберите хотя бы одну модель', style={'color': 'red'})

    try:
        df_raw = tsm_db.prepare_dataframe_from_db(
            db,
            source=DEFAULT_SOURCE,
            lex_mode=lex_mode,
            sem_mode=sem_mode,
            rouge_measure=rouge_measure,
            bertscore_measure=bertscore_measure or 'f',
            models=selected_models,
            model_label='name_version',
            verbose=False,
        )
        df, cal, thr = eng.run_pipeline_from_df(
            df_raw,
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
        html.Span(f'Моделей: {df["model"].nunique()}'),
    ])

    return df.to_json(orient='records'), cal, thr, stats


# ═══════════════════════════════════════════════════════════════════════
# Callback 2: Отрисовка 2D графиков (Вкладка 1)
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
            style={'height': '350px', 'border': '1px solid #ddd'},
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
# Callback 3: Клик по точке → выбор документа (Вкладка 1)
# ═══════════════════════════════════════════════════════════════════════

@callback(
    Output('store-selected-doc', 'data'),
    Input({'type': 'graph-2d', 'index': ALL}, 'clickData'),
    prevent_initial_call=True,
)
def on_click(click_data_list):
    for cd in click_data_list:
        if cd is not None:
            point = cd['points'][0]
            if 'customdata' in point:
                return int(point['customdata'][0])
    return no_update


# ═══════════════════════════════════════════════════════════════════════
# Callback 4: Инфо-панель (Вкладка 1)
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

    # Загрузить тексты из БД
    pub_id = int(sub.iloc[0]['publication_id'])
    first_model = sub.iloc[0]['model']
    texts = tsm_db.load_texts_from_db(db, pub_id, first_model)

    children = [
        html.H4(f'Документ #{selected_doc} (pub_id={pub_id})', style={'margin': '0 0 10px 0', 'color': '#2c3e50'}),
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
        children.append(_collapsible_text_block(
            title='Авторский реферат',
            text=target,
            border_color='#2c3e50',
            bg_color='#f0f0f0',
        ))

    # Рефераты моделей
    if len(sub) > 0:
        best = sub.iloc[0]
        worst = sub.iloc[-1]

        # Лучший и худший
        for label, row in [('Лучший', best), ('Худший', worst)]:
            model_col = row['model']
            model_texts = tsm_db.load_texts_from_db(db, pub_id, model_col)
            model_text = model_texts.get('model_summary', '')
            if not model_text:
                continue
            diag = row['diagnosis_type']
            color = eng.DIAGNOSIS_COLORS.get(diag, '#ccc')
            children.append(_collapsible_text_block(
                title=f'{label}: {MODEL_SHORT.get(model_col, model_col)}',
                subtitle=f'{eng.DIAGNOSIS_LABELS_RU.get(diag, diag)} (Q={row["Q"]:.3f})',
                text=model_text,
                border_color=color,
                bg_color=f'{color}11',
                open_default=True,
            ))

        # Остальные модели
        shown_models = {best['model'], worst['model']}
        rest = sub[~sub['model'].isin(shown_models)]
        if len(rest) > 0:
            rest_blocks = []
            for _, row in rest.iterrows():
                model_col = row['model']
                model_texts = tsm_db.load_texts_from_db(db, pub_id, model_col)
                model_text = model_texts.get('model_summary', '')
                if not model_text:
                    continue
                diag = row['diagnosis_type']
                color = eng.DIAGNOSIS_COLORS.get(diag, '#ccc')
                rest_blocks.append(_collapsible_text_block(
                    title=MODEL_SHORT.get(model_col, model_col),
                    subtitle=f'{eng.DIAGNOSIS_LABELS_RU.get(diag, diag)} (Q={row["Q"]:.3f})',
                    text=model_text,
                    border_color=color,
                    bg_color=f'{color}11',
                    open_default=False,
                ))

            if rest_blocks:
                children.append(html.Details([
                    html.Summary(f'Остальные модели ({len(rest_blocks)})',
                                 style={'cursor': 'pointer', 'fontWeight': 'bold',
                                        'fontSize': '13px', 'padding': '6px 0',
                                        'color': '#2c3e50'}),
                    html.Div(rest_blocks, style={'paddingTop': '6px'}),
                ], open=False, style={'marginBottom': '10px',
                                      'border': '1px solid #ddd', 'borderRadius': '4px',
                                      'padding': '6px 10px'}))

    return children


def _collapsible_text_block(title, text, border_color='#ccc', bg_color='#f8f8f8',
                            subtitle=None, open_default=True, preview_len=400):
    """Блок текста со сворачиванием, если текст длинный.

    Короткий текст (≤ preview_len) показывается целиком.
    Длинный — обрезанное превью + <details> с полным текстом.
    """
    title_parts = [html.Strong(title, style={'color': border_color})]
    if subtitle:
        title_parts.append(html.Span(f' — {subtitle}',
                                     style={'fontSize': '11px', 'color': '#666'}))
    title_parts.append(html.Span(f' ({len(text)} симв.)',
                                 style={'color': '#999', 'fontSize': '11px'}))

    text_style = {
        'padding': '6px', 'background': bg_color,
        'borderLeft': f'3px solid {border_color}', 'borderRadius': '4px',
        'fontSize': '12px', 'marginTop': '4px', 'whiteSpace': 'pre-wrap',
    }

    if len(text) <= preview_len:
        body = html.Div(text, style=text_style)
    else:
        body = html.Div([
            html.Div(text[:preview_len] + '…', style=text_style),
            html.Details([
                html.Summary('Развернуть полностью',
                             style={'cursor': 'pointer', 'fontSize': '11px',
                                    'color': '#3498db', 'padding': '4px 0',
                                    'userSelect': 'none'}),
                html.Div(text, style=text_style),
            ], open=False),
        ])

    return html.Div([html.Div(title_parts), body], style={'marginBottom': '10px'})


# ═══════════════════════════════════════════════════════════════════════
# Callback 5: 3D график (Вкладка 1)
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

    # Scatter3d поддерживает: circle, circle-open, cross, diamond, diamond-open, square, square-open, x
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

    # Пороговый параллелепипед
    if thresholds:
        tll = thresholds.get('tau_lex_lower', -2)
        tlu = thresholds.get('tau_lex_upper', 2)
        tsl = thresholds.get('tau_sem_lower', -2)
        tsu = thresholds.get('tau_sem_upper', 2)
        tcl = thresholds.get('tau_comp_lower', -2)
        tcu = thresholds.get('tau_comp_upper', 2)

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

    return [dcc.Graph(figure=fig, style={'height': '550px'})], {'display': 'block', 'margin-bottom': '15px', 'border': '1px solid #ddd'}


# ═══════════════════════════════════════════════════════════════════════
# Callback 6: Вычисление метрик ручного ввода (Вкладка 2)
# ═══════════════════════════════════════════════════════════════════════

@callback(
    Output('div-manual-results', 'children'),
    Input('btn-compute-manual', 'n_clicks'),
    State('textarea-source', 'value'),
    State('textarea-summary', 'value'),
    State('textarea-reference', 'value'),
    State('store-calibration', 'data'),
    State('store-thresholds', 'data'),
    State('store-results', 'data'),
    State('dd-lex-metric', 'value'),
    State('dd-sem-metric', 'value'),
    State('radio-rouge-measure', 'value'),
    State('radio-bertscore-measure', 'value'),
    State('input-alpha', 'value'),
    State('input-beta', 'value'),
    State('input-gamma', 'value'),
    State('input-delta', 'value'),
    prevent_initial_call=True,
)
def compute_manual_metrics(n_clicks, source_text, summary_text, reference_text,
                           calibration, thresholds, results_json,
                           lex_mode, sem_mode, rouge_measure, bertscore_measure,
                           alpha, beta, gamma, delta):
    if not source_text or not source_text.strip():
        return html.Div('Введите исходный текст.', style={'color': '#e74c3c', 'fontWeight': 'bold'})
    if not summary_text or not summary_text.strip():
        return html.Div('Введите реферат.', style={'color': '#e74c3c', 'fontWeight': 'bold'})
    if calibration is None or thresholds is None:
        return html.Div(
            'Сначала запустите анализ на вкладке «Анализ моделей» (нажмите «Применить»).',
            style={'color': '#e74c3c', 'fontWeight': 'bold'},
        )

    # Embedding-метрики не поддерживаются для ручного ввода
    if sem_mode and sem_mode.startswith('emb:'):
        return html.Div(
            'Embedding-метрики не поддерживаются для ручного ввода. '
            'Выберите BERTScore на вкладке «Анализ моделей».',
            style={'color': '#e74c3c'},
        )

    source_text = source_text.strip()
    summary_text = summary_text.strip()
    reference_text = reference_text.strip() if reference_text else None

    # 1. Вычислить метрики
    try:
        all_metrics = mc.compute_all_metrics(
            source=source_text,
            summary=summary_text,
            reference=reference_text,
            device='cuda',
        )
    except Exception as e:
        return html.Div(f'Ошибка вычисления метрик: {e}', style={'color': '#e74c3c'})

    # 2. Оценить через TSM
    try:
        result = eng.evaluate_manual_input(
            metrics_dict=all_metrics,
            calibration=calibration,
            thresholds=thresholds,
            lex_mode=lex_mode or 'rouge1',
            sem_mode=sem_mode or 'bertscore',
            rouge_measure=rouge_measure or 'p',
            bertscore_measure=bertscore_measure or 'f',
            alpha=alpha or 0.45,
            beta=beta or 0.25,
            gamma=gamma or 0.15,
            delta=delta or 0.15,
        )
    except Exception as e:
        return html.Div(f'Ошибка оценки: {e}', style={'color': '#e74c3c'})

    if 'error' in result:
        return html.Div(f'Ошибка: {result["error"]}', style={'color': '#e74c3c'})

    # 3. Собрать UI результатов
    return _build_manual_results_ui(result, all_metrics, thresholds, results_json,
                                    lex_mode, sem_mode, rouge_measure)


def _build_manual_results_ui(result, all_metrics, thresholds, results_json,
                              lex_mode, sem_mode, rouge_measure):
    """Собрать HTML-контент с результатами ручного анализа."""
    diag = result['diagnosis_type']
    diag_color = eng.DIAGNOSIS_COLORS.get(diag, '#ccc')
    diag_label = eng.DIAGNOSIS_LABELS_RU.get(diag, diag)
    has_ref = result['has_reference']
    lengths = all_metrics.get('lengths', {})

    children = []

    # Строка длин текстов
    length_items = [
        html.Span(f'ОТ: {lengths.get("source", 0)} симв.', style={'fontSize': '12px'}),
        html.Span(' | '),
        html.Span(f'СР: {lengths.get("summary", 0)} симв.', style={'fontSize': '12px'}),
        html.Span(' | '),
        html.Span(f'Сжатие: {all_metrics.get("compression_ratio", 0):.3f}',
                  style={'fontSize': '12px'}),
    ]
    if lengths.get('reference'):
        length_items += [
            html.Span(' | '),
            html.Span(f'АР: {lengths["reference"]} симв.', style={'fontSize': '12px'}),
        ]

    # ── Верхняя часть: карточка + график ──
    children.append(html.Div([
        # ЛЕВАЯ КОЛОНКА: Карточка диагноза + таблица метрик
        html.Div([
            # Диагноз
            html.Div([
                html.Div([
                    html.Span('Диагноз: ', style={'fontSize': '16px', 'fontWeight': 'bold'}),
                    html.Span(diag_label, style={
                        'fontSize': '18px', 'fontWeight': 'bold', 'color': 'white',
                        'backgroundColor': diag_color, 'padding': '4px 12px',
                        'borderRadius': '4px',
                    }),
                ], style={'marginBottom': '10px'}),

                html.Div([
                    html.Span(f'Уверенность: {result["diagnosis_confidence"]:.2f}',
                              style={'fontSize': '13px', 'color': '#666'}),
                ], style={'marginBottom': '8px'}),

                # Q-score
                html.Div([
                    html.Span('Q-score: ', style={'fontSize': '15px', 'fontWeight': 'bold'}),
                    html.Span(f'{result["Q"]:.4f}', style={
                        'fontSize': '20px', 'fontWeight': 'bold', 'color': '#2c3e50',
                    }),
                    html.Span(' (упрощ.)' if not has_ref else '',
                              style={'fontSize': '11px', 'color': '#999', 'marginLeft': '4px'}),
                ], style={'marginBottom': '12px'}),

                # Z-scores
                html.Div([
                    _z_badge('z_lex', result['z_lex']),
                    _z_badge('z_sem', result['z_sem']),
                    _z_badge('z_comp', result['z_comp']),
                ], style={'display': 'flex', 'gap': '8px', 'marginBottom': '12px'}),

                # Компоненты Q
                html.Div([
                    html.Strong('Компоненты Q:', style={'fontSize': '12px'}),
                    html.Div([
                        html.Span(f'q_sem={result["q_sem"]:.3f} (×α)', style={'fontSize': '12px'}),
                        html.Span(' | ', style={'color': '#ddd'}),
                        html.Span(f'q_lex={result["q_lex"]:.3f} (×β)', style={'fontSize': '12px'}),
                        html.Span(' | ', style={'color': '#ddd'}),
                        html.Span(f'q_align={result["q_align"]:.3f} (×γ)', style={'fontSize': '12px'}),
                        html.Span(' | ', style={'color': '#ddd'}),
                        html.Span(f'q_comp={result["q_comp"]:.3f} (×δ)', style={'fontSize': '12px'}),
                    ]),
                ], style={'padding': '6px', 'background': '#f8f9fa', 'borderRadius': '4px',
                          'marginBottom': '12px'}),

                # Длины текстов
                html.Div(length_items, style={'marginBottom': '12px', 'color': '#666'}),

            ], style={'padding': '12px', 'border': f'2px solid {diag_color}',
                      'borderRadius': '8px', 'marginBottom': '12px'}),

            # Таблица всех метрик
            _build_metrics_table(all_metrics),

        ], style={'width': '380px', 'minWidth': '340px'}),

        # ПРАВАЯ КОЛОНКА: 2D + 3D scatter
        html.Div([
            dcc.Graph(
                id='graph-manual-scatter',
                figure=_make_manual_scatter(result, thresholds, results_json),
                config={'displayModeBar': True, 'displaylogo': False},
                style={'height': '500px'},
            ),
            dcc.Graph(
                id='graph-manual-3d',
                figure=_make_manual_scatter_3d(result, thresholds, results_json),
                config={'displayModeBar': True, 'displaylogo': False},
                style={'height': '550px', 'marginTop': '10px', 'border': '1px solid #ddd'},
            ),
        ], style={'flex': '1', 'minWidth': '400px'}),

    ], style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap'}))

    return children


def _z_badge(label, value):
    """Плашка со z-score."""
    color = '#2ecc71' if abs(value) < 1 else '#f39c12' if abs(value) < 2 else '#e74c3c'
    return html.Span(
        f'{label} = {value:+.3f}',
        style={
            'fontSize': '13px', 'fontWeight': 'bold', 'padding': '3px 8px',
            'borderRadius': '4px', 'border': f'1px solid {color}', 'color': color,
        },
    )


def _build_metrics_table(all_metrics):
    """Таблица всех вычисленных метрик."""
    ot_sr = all_metrics.get('ot_sr', {})
    ar_sr = all_metrics.get('ar_sr')

    rows = [html.Tr([
        html.Th('Метрика', style={'padding': '4px 8px', 'textAlign': 'left'}),
        html.Th('ОТ-СР', style={'padding': '4px 8px', 'textAlign': 'center'}),
        *([html.Th('АР-СР', style={'padding': '4px 8px', 'textAlign': 'center'})] if ar_sr else []),
    ], style={'background': '#2c3e50', 'color': 'white', 'fontSize': '11px'})]

    def _fmt(val):
        if val is None:
            return 'н/д'
        return f'{val:.4f}'

    # ROUGE
    rouge = ot_sr.get('rouge', {})
    rouge_ar = (ar_sr or {}).get('rouge', {})
    for rtype in ('rouge1', 'rouge2', 'rougeL'):
        for measure, mlabel in [('p', 'P'), ('r', 'R'), ('f', 'F')]:
            val = rouge.get(rtype, {}).get(measure)
            val_ar = rouge_ar.get(rtype, {}).get(measure) if ar_sr else None
            rows.append(html.Tr([
                html.Td(f'{rtype.upper()} {mlabel}', style={'padding': '2px 8px', 'fontSize': '12px'}),
                html.Td(_fmt(val), style={'padding': '2px 8px', 'textAlign': 'center', 'fontSize': '12px'}),
                *([html.Td(_fmt(val_ar), style={'padding': '2px 8px', 'textAlign': 'center', 'fontSize': '12px'})] if ar_sr else []),
            ], style={'borderBottom': '1px solid #eee'}))

    # BLEU, chrF, METEOR
    for metric_key, metric_label in [('bleu', 'BLEU'), ('chrf', 'chrF++'), ('meteor', 'METEOR')]:
        val = ot_sr.get(metric_key)
        val_ar = (ar_sr or {}).get(metric_key) if ar_sr else None
        rows.append(html.Tr([
            html.Td(metric_label, style={'padding': '2px 8px', 'fontSize': '12px', 'fontWeight': 'bold'}),
            html.Td(_fmt(val), style={'padding': '2px 8px', 'textAlign': 'center', 'fontSize': '12px'}),
            *([html.Td(_fmt(val_ar), style={'padding': '2px 8px', 'textAlign': 'center', 'fontSize': '12px'})] if ar_sr else []),
        ], style={'borderBottom': '1px solid #ddd', 'background': '#f8f9fa'}))

    # BERTScore
    bs = ot_sr.get('bertscore')
    bs_ar = (ar_sr or {}).get('bertscore') if ar_sr else None
    if bs is not None:
        for key, label in [('precision', 'P'), ('recall', 'R'), ('f1', 'F1')]:
            val = bs.get(key) if isinstance(bs, dict) else None
            val_ar = bs_ar.get(key) if isinstance(bs_ar, dict) else None
            rows.append(html.Tr([
                html.Td(f'BERTScore {label}', style={'padding': '2px 8px', 'fontSize': '12px'}),
                html.Td(_fmt(val), style={'padding': '2px 8px', 'textAlign': 'center', 'fontSize': '12px'}),
                *([html.Td(_fmt(val_ar), style={'padding': '2px 8px', 'textAlign': 'center', 'fontSize': '12px'})] if ar_sr else []),
            ], style={'borderBottom': '1px solid #eee'}))
    else:
        rows.append(html.Tr([
            html.Td('BERTScore', style={'padding': '2px 8px', 'fontSize': '12px'}),
            html.Td('н/д', style={'padding': '2px 8px', 'textAlign': 'center', 'fontSize': '12px', 'color': '#999'}),
            *([html.Td('н/д', style={'padding': '2px 8px', 'textAlign': 'center', 'fontSize': '12px', 'color': '#999'})] if ar_sr else []),
        ], style={'borderBottom': '1px solid #eee'}))

    # BLEURT
    bleurt_val = ot_sr.get('bleurt')
    bleurt_ar = (ar_sr or {}).get('bleurt') if ar_sr else None
    rows.append(html.Tr([
        html.Td('BLEURT', style={'padding': '2px 8px', 'fontSize': '12px', 'fontWeight': 'bold'}),
        html.Td(_fmt(bleurt_val), style={'padding': '2px 8px', 'textAlign': 'center', 'fontSize': '12px'}),
        *([html.Td(_fmt(bleurt_ar), style={'padding': '2px 8px', 'textAlign': 'center', 'fontSize': '12px'})] if ar_sr else []),
    ], style={'borderBottom': '1px solid #ddd', 'background': '#f8f9fa'}))

    # Сжатие
    comp = all_metrics.get('compression_ratio')
    rows.append(html.Tr([
        html.Td('Сжатие', style={'padding': '2px 8px', 'fontSize': '12px', 'fontWeight': 'bold'}),
        html.Td(_fmt(comp), style={'padding': '2px 8px', 'textAlign': 'center', 'fontSize': '12px'}),
        *([html.Td('—', style={'padding': '2px 8px', 'textAlign': 'center', 'fontSize': '12px'})] if ar_sr else []),
    ], style={'background': '#f8f9fa'}))

    return html.Div([
        html.Strong('Все метрики', style={'fontSize': '13px', 'marginBottom': '6px', 'display': 'block'}),
        html.Table(rows, style={
            'borderCollapse': 'collapse', 'width': '100%', 'border': '1px solid #ddd',
        }),
    ])


def _make_manual_scatter(result, thresholds, results_json):
    """Scatter plot: фон моделей + точка ручного реферата."""
    fig = go.Figure()

    # Пороговая зона
    if thresholds:
        tll = thresholds.get('tau_lex_lower', -2)
        tlu = thresholds.get('tau_lex_upper', 2)
        tsl = thresholds.get('tau_sem_lower', -2)
        tsu = thresholds.get('tau_sem_upper', 2)
        fig.add_shape(type='rect', x0=tll, x1=tlu, y0=tsl, y1=tsu,
                      fillcolor='rgba(46,204,113,0.12)', line=dict(color='#2ecc71', width=1.5, dash='dash'))

    # Фоновые точки моделей
    if results_json:
        try:
            df = pd.read_json(results_json, orient='records')
            fig.add_trace(go.Scatter(
                x=df['z_lex'], y=df['z_sem'],
                mode='markers',
                marker=dict(
                    color=[eng.DIAGNOSIS_COLORS.get(d, '#ccc') for d in df['diagnosis_type']],
                    size=4, opacity=0.15,
                ),
                name='Модели',
                hoverinfo='skip',
                showlegend=True,
            ))
        except Exception:
            pass

    # Точка ручного реферата
    diag = result['diagnosis_type']
    diag_color = eng.DIAGNOSIS_COLORS.get(diag, '#ccc')
    diag_label = eng.DIAGNOSIS_LABELS_RU.get(diag, diag)

    fig.add_trace(go.Scatter(
        x=[result['z_lex']], y=[result['z_sem']],
        mode='markers+text',
        marker=dict(color=diag_color, size=18, symbol='star',
                    line=dict(width=2, color='#2c3e50')),
        text=[f'  {diag_label}'],
        textposition='middle right',
        textfont=dict(size=12, color=diag_color),
        name='Ручной реферат',
        hovertemplate=(
            f'Ручной реферат<br>'
            f'z_lex=%{{x:.3f}}<br>'
            f'z_sem=%{{y:.3f}}<br>'
            f'z_comp={result["z_comp"]:.3f}<br>'
            f'Q={result["Q"]:.4f}<br>'
            f'{diag_label}<extra></extra>'
        ),
        showlegend=True,
    ))

    fig.update_layout(
        title=dict(text='Позиция реферата среди моделей', font=dict(size=14)),
        xaxis=dict(title='z_lex (лексическое отклонение)', zeroline=True, zerolinewidth=0.5),
        yaxis=dict(title='z_sem (семантическое отклонение)', zeroline=True, zerolinewidth=0.5),
        margin=dict(l=50, r=20, t=40, b=40),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=500,
        legend=dict(
            yanchor='top', y=0.99, xanchor='left', x=0.01,
            bgcolor='rgba(255,255,255,0.8)', font=dict(size=11),
        ),
    )
    fig.update_xaxes(gridcolor='#eee', gridwidth=0.5)
    fig.update_yaxes(gridcolor='#eee', gridwidth=0.5)

    return fig


def _make_manual_scatter_3d(result, thresholds, results_json):
    """3D scatter: фон моделей + точка ручного реферата (z_lex × z_sem × z_comp)."""
    fig = go.Figure()

    # Фоновые точки моделей
    if results_json:
        try:
            df = pd.read_json(results_json, orient='records')
            symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'circle-open',
                       'square-open', 'diamond-open']
            models = sorted(df['model'].unique())

            for i, model in enumerate(models):
                sub = df[df['model'] == model]
                model_label = MODEL_SHORT.get(model, model)
                fig.add_trace(go.Scatter3d(
                    x=sub['z_lex'], y=sub['z_sem'], z=sub['z_comp'],
                    mode='markers',
                    marker=dict(
                        color=[eng.DIAGNOSIS_COLORS.get(d, '#ccc') for d in sub['diagnosis_type']],
                        size=3, opacity=0.2,
                        symbol=symbols[i % len(symbols)],
                    ),
                    name=model_label,
                    hoverinfo='skip',
                ))
        except Exception:
            pass

    # Пороговый параллелепипед
    if thresholds:
        tll = thresholds.get('tau_lex_lower', -2)
        tlu = thresholds.get('tau_lex_upper', 2)
        tsl = thresholds.get('tau_sem_lower', -2)
        tsu = thresholds.get('tau_sem_upper', 2)
        tcl = thresholds.get('tau_comp_lower', -2)
        tcu = thresholds.get('tau_comp_upper', 2)

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

    # Точка ручного реферата
    diag = result['diagnosis_type']
    diag_color = eng.DIAGNOSIS_COLORS.get(diag, '#ccc')
    diag_label = eng.DIAGNOSIS_LABELS_RU.get(diag, diag)

    fig.add_trace(go.Scatter3d(
        x=[result['z_lex']], y=[result['z_sem']], z=[result['z_comp']],
        mode='markers+text',
        marker=dict(color=diag_color, size=10, symbol='diamond',
                    line=dict(width=2, color='#2c3e50')),
        text=[diag_label],
        textposition='top center',
        textfont=dict(size=12, color=diag_color),
        name='Ручной реферат',
        hovertemplate=(
            f'Ручной реферат<br>'
            f'z_lex=%{{x:.3f}}<br>'
            f'z_sem=%{{y:.3f}}<br>'
            f'z_comp=%{{z:.3f}}<br>'
            f'Q={result["Q"]:.4f}<br>'
            f'{diag_label}<extra></extra>'
        ),
        showlegend=True,
    ))

    fig.update_layout(
        title=dict(text='3D: позиция реферата (z_lex × z_sem × z_comp)', font=dict(size=13)),
        scene=dict(
            xaxis_title='z_lex', yaxis_title='z_sem', zaxis_title='z_comp',
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
        ),
        margin=dict(l=0, r=0, t=35, b=0),
        height=550,
        legend=dict(font=dict(size=11)),
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════
# Автозапуск при старте
# ═══════════════════════════════════════════════════════════════════════

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
