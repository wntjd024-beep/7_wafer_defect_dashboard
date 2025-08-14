import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import joblib
model_data = joblib.load("defect_models.pkl")
models = model_data["models"]
feature_m = model_data["features"]
feature_ranges = model_data["feature_ranges"]

# ---------------------- 데이터 불러오기 ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("basic_raw_data(원본).csv")  # 파일명은 그대로 유지
    return df
df = load_data()
df['Class'] = df['Class'].astype(str)
# ---------------------- 페이지 설정 ----------------------
st.set_page_config(page_title="Wafer Defect Dashboard", layout="wide")

# ---------------------- 네비게이션 ----------------------
with st.sidebar:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("sparta.png", width=150)

    selected = option_menu(
        "메뉴", 
        ["Wafer Map", "결함 통계", "머신러닝 시뮬레이터"],
        icons=["map", "bar-chart", "cpu"], menu_icon="cast", default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#f9f9f9"},
            "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px",
                         "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "salmon"}
        }
    )

# ---------------------- 필터 UI ----------------------
with st.sidebar:
    st.markdown("## 필터")
    df_filtered = df[df["Class"] != "9"]
    
    step_options = sorted(df_filtered["Step_desc"].dropna().unique())
    selected_steps = st.multiselect("공정 (Step)", options=step_options)
    if selected_steps:
        df_filtered = df_filtered[df_filtered["Step_desc"].isin(selected_steps)]
    class_options = sorted(df_filtered["Class"].dropna().unique())
    selected_classes = st.multiselect("결함 Class", options=class_options)
    if selected_classes:
        df_filtered = df_filtered[df_filtered["Class"].isin(selected_classes)]
    slot_options = sorted(df_filtered["Slot No"].dropna().unique())
    selected_slots = st.multiselect("슬롯 번호 (Slot No)", options=slot_options)
    if selected_slots:
        df_filtered = df_filtered[df_filtered["Slot No"].isin(selected_slots)]
# ---------------------- 필터 적용 ----------------------
filtered_df = df.copy()
if selected_steps:
    filtered_df = filtered_df[filtered_df["Step_desc"].isin(selected_steps)]
if selected_classes:
    filtered_df = filtered_df[filtered_df["Class"].isin(selected_classes)]
if selected_slots:
    filtered_df = filtered_df[filtered_df["Slot No"].isin(selected_slots)]
# ---------------------- Wafer Map ----------------------
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go

if selected == "Wafer Map":
    st.markdown(
        """
        <h1 style="
            color: black; 
            font-size: 48px; 
            font-weight: 600; 
            text-align: center;
            ">
            반도체 결함 대시보드
        </h1>
        """, 
        unsafe_allow_html=True
    )
    seperaor = ", "
    st.markdown("----------------------------------------------------")
    st.markdown(f"<h2 style='font-size:32px;'>{seperaor.join(selected_steps)} 주요 KPI</h2>", unsafe_allow_html=True)

    def kpi_card(title, value, unit=""):
        return f"""
        <div style='background-color:#fff5f5; padding:15px; border-radius:10px; text-align:center;'>
            <p style='margin:0; font-size:20px; color:gray;'>{title}</p>
            <h1 style='margin:0; font-size:36px; font-weight:bold; color:salmon;'>{value}{unit}</h1>
        </div>
        """

    # KPI 값 계산
    total_defects = filtered_df['Step_desc'].count()
    defect_rate = round(100 * len(filtered_df[filtered_df['Class'] != '9']) / len(filtered_df), 1)
    top_class = filtered_df[filtered_df["Class"] != "9"]['Class'].value_counts().idxmax()
    top_slot = filtered_df[filtered_df["Class"] != "9"]['Slot No'].value_counts().idxmax()

    # KPI 표시
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(kpi_card("총 결함 수", f"{total_defects:,}"), unsafe_allow_html=True)
    col2.markdown(kpi_card("불량률 (%)", defect_rate, "%"), unsafe_allow_html=True)
    col3.markdown(kpi_card("TOP 불량 Class", f"Class {top_class}"), unsafe_allow_html=True)
    col4.markdown(kpi_card("TOP 불량 Slot", f"Slot {top_slot}"), unsafe_allow_html=True)
    st.subheader("")
    if (selected_steps == []) & (selected_classes == []) & (selected_slots == []):
        def make_gauge(title, value):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                number={'suffix': "%", 'font': {'size': 50}},
                gauge={
                    'axis': {'range': [0, 100], 'tickvals': [25, 50, 75]},  # 눈금 변경
                    'bar': {'color': "salmon"}, 
                    'bgcolor': "white",
                    'steps': [],  # 단계 색상 제거
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.6,
                        'value': value
                    }
                },
                title={'text': title, 'font': {'size': 30}}
            ))
            fig.update_layout(height=250, margin=dict(t=80, b=0, l=0, r=0))
            return fig
        
        st.markdown("----------------------------------------------------")
        st.markdown(f"<h2 style='font-size:32px;'>{seperaor.join(selected_steps)} 공정별 불량률 현황</h2>", unsafe_allow_html=True)

        col_g1, col_g2, col_g3 = st.columns(3)
        for col, step_name in zip([col_g1, col_g2, col_g3], ["CBCMP", "PC", "RMG"]):
            step_df = df[df["Step_desc"] == step_name]
            if not step_df.empty:
                defect_rate = round(100 * len(step_df[step_df["Class"] != "9"]) / len(step_df), 1)
                col.plotly_chart(make_gauge(step_name, defect_rate), use_container_width=True)
            else:
                col.write(f"{step_name} 데이터 없음")
    else:
        st.markdown("---------------------------------")
        col1, col2= st.columns(2)
        with col1:
            theta = filtered_df['ANGLE']
            rad = np.deg2rad(theta)
            filtered_df["X"] = filtered_df["RADIUS"] * np.cos(rad)/1000
            filtered_df["Y"] = filtered_df["RADIUS"] * np.sin(rad)/1000
            wafer_radius = filtered_df["RADIUS"].max() /1000
            theta = np.linspace(0, 2*np.pi, 500)
            wafer_x = wafer_radius * np.cos(theta)
            wafer_y = wafer_radius * np.sin(theta)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=wafer_x,
                y=wafer_y,
                mode='lines',
                line=dict(color='black', width=2),
                name='Wafer Edge',
                hoverinfo='skip'
            ))
            fig.add_trace(go.Scattergl(
                x=filtered_df["X"],
                y=filtered_df["Y"],
                mode='markers',
                marker=dict(
                    color='red',
                    size=2,
                    opacity=0.7,
                    line=dict(width=0.1, color='black')
                ),
                text=filtered_df["Class"],
                hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<br>Class: %{text}<extra></extra>',
                name='Defects'
            ))
            fig.update_layout(
                width=400,
                height=500,
                xaxis=dict(scaleanchor="y", title='X', range=[-wafer_radius, wafer_radius]),
                yaxis=dict(title='Y', range=[-wafer_radius, wafer_radius]),
                title={
                    'text': "💿 Wafer Map",
                    'x': 0.05,
                    'y': 0.9, 
                    'xanchor': 'left',
                    'yanchor': 'top',
                    'font': {'size': 24, 'color': 'black', 'family': "Arial"}
                },
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            class_summary = (
                filtered_df[filtered_df["Class"]!="9"].groupby("Class")
                .size()
                .reset_index(name="Count")
                .sort_values(by="Count", ascending=False)
            )
            if len(selected_classes) == 1:
                class_val = selected_classes[0]
                class_df = filtered_df[filtered_df["Class"] == class_val]
                with st.container(height=600, border=True):
                    st.markdown(
                        f"<h3 style='text-align:left;'>📌 선택한 Class {class_val} 상세 정보</h3>",
                        unsafe_allow_html=True
                    )
                    st.markdown(" ")
                    with st.container(height =200, border =False):
                        st.markdown("#### 공정별 결함 건수")
                        st.markdown(" ")
                        st.markdown(" ")
                        st.markdown(" ")
                        step_summary = class_df["Step_desc"].value_counts().reset_index()
                        step_summary.columns = ["Step_desc", "Count"]
                        _, center, _ = st.columns([2, 6, 2])
                        center.dataframe(step_summary, hide_index = True)
                    st.markdown("--------------------")
                    st.markdown("#### 중요 변수 평균값")
                    st.markdown("""
                    <style>
                    .metric-box {
                        background-color: #F5F5F5; /* 배경 */
                        border-radius: 10px;
                        padding: 15px;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        height: 100px;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.05); /* 살짝 그림자 */
                    }
                    .metric-value {
                        font-size: 24px;
                        font-weight: bold;
                        text-align: center;
                        color: #333;
                    }
                    .metric-label {
                        font-size: 14px;
                        color: gray;
                        text-align: center;
                    }
                    </style>
                    """, unsafe_allow_html=True)

                    col_a, col_b, col_c, col_d, col_e = st.columns(5)

                    with col_a:
                        st.markdown(f"<div class='metric-box'><div class='metric-label'>SIZE_X</div><div class='metric-value'>{round(class_df['SIZE_X'].mean(), 2)}</div></div>", unsafe_allow_html=True)

                    with col_b:
                        st.markdown(f"<div class='metric-box'><div class='metric-label'>POLARITY</div><div class='metric-value'>{round(class_df['POLARITY'].mean(), 2)}</div></div>", unsafe_allow_html=True)

                    with col_c:
                        st.markdown(f"<div class='metric-box'><div class='metric-label'>INTENSITY</div><div class='metric-value'>{round(class_df['INTENSITY'].mean(), 2)}</div></div>", unsafe_allow_html=True)

                    with col_d:
                        st.markdown(f"<div class='metric-box'><div class='metric-label'>ACTIVERATIO</div><div class='metric-value'>{round(class_df['ACTIVERATIO'].mean(), 2)}</div></div>", unsafe_allow_html=True)

                    with col_e:
                        st.markdown(f"<div class='metric-box'><div class='metric-label'>SPOTLIKENESS</div><div class='metric-value'>{round(class_df['SPOTLIKENESS'].mean(), 2)}</div></div>", unsafe_allow_html=True)
            else:
                class_summary['label'] = class_summary['Class'].astype(str).apply(lambda x: f"Class {x}")
                # go.Bar 생성
                fig_bar = go.Figure(go.Bar(
                    x=class_summary['Count'],
                    y=class_summary['label'],
                    orientation='h',
                    text=class_summary['Count'],
                    textposition='auto',
                    marker=dict(color='salmon')
                ))
                fig_bar.update_layout(
                    yaxis=dict(
                        categoryorder='array',
                        categoryarray=class_summary['label'].tolist(),
                        autorange='reversed',
                        title_font=dict(size=20),  # y축 제목 폰트 크기
                        tickfont=dict(size=16)  
                    ),
                    height=500,
                    margin=dict(l=100, r=50, t=120, b=0),
                    title=dict(
                    text="📊 Class별 결함 개수",
                    font=dict(size=24) 
                )
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
# ---------------------- 결함 통계 ----------------------
# ---------------------- 결함 통계(공정별) ----------------------
elif selected == "결함 통계":
    st.markdown(
        """
        <h1 style="
            color: black; 
            font-size: 48px; 
            font-weight: 600; 
            text-align: center;
            ">
            공정별 / 슬롯별 결함률 분석
        </h1>
        """, 
        unsafe_allow_html=True
    )
    st.markdown("----------------------------------------------------")
    col1, col_sep, col2 = st.columns([5, 0.1, 5])
    # ---------------------- 공정별 결함 건수 ----------------------
    with col1:
        st.subheader("공정별 결함 건수")
        step_count = filtered_df["Step_desc"].value_counts().sort_index().reset_index()
        step_count.columns = ["Step_desc", "Count"]
        fig_step = px.bar(
            step_count,
            x="Step_desc",
            y="Count",
            text="Count",
            color="Count",
            color_continuous_scale="Peach"
        )
        fig_step.update_traces(
        textposition="inside",
        insidetextfont=dict(size=20, color="black") 
        )
        
        fig_step.update_layout(
            xaxis_title="공정명",
            yaxis_title="결함 건수",
            showlegend=False,
            shapes=[dict(
                type="rect", xref="paper", yref="paper",
                x0=0, y0=0, x1=1, y1=1,
                line=dict(color="lightgray", width=1, dash="dash")
            )]
        )
        fig_step.update_coloraxes(showscale=False)
        st.plotly_chart(fig_step, use_container_width=True)
    with col_sep:
        st.markdown(
            """
            <div style="border-left:1px solid lightgray; height:600px; margin-left:10px;"></div>
            """,
            unsafe_allow_html=True
        )
    # ---------------------- 슬롯별 결함 건수 ----------------------
    with col2:
        st.subheader("슬롯별 결함 건수")
        slot_count = filtered_df["Slot No"].value_counts().sort_index().reset_index()
        slot_count.columns = ["Slot No", "Count"]
        fig_slot = px.bar(
            slot_count,
            x="Slot No",
            y="Count",
            text="Count",
            color="Count",
            color_continuous_scale="Peach"
        )
        fig_slot.update_traces(textposition="outside")
        fig_slot.update_layout(
            xaxis_title="슬롯 번호",
            yaxis_title="결함 건수",
            showlegend=False,
            shapes=[dict(
                type="rect", xref="paper", yref="paper",
                x0=0, y0=0, x1=1, y1=1,
                line=dict(color="lightgray", width=1, dash="dash")
            )]
        )
        fig_slot.update_coloraxes(showscale=False)
        st.plotly_chart(fig_slot, use_container_width=True)
    # ---------------------- 클래스별 분포 ----------------------
    st.markdown("---")
    st.subheader("클래스별 분포")
    filtered_df["Class"] = filtered_df["Class"].astype(int)
    class_order = [9, 10, 14, 17, 20, 21, 22, 28, 39, 56, 99]
    class_count = (
        filtered_df["Class"]
        .value_counts()
        .reindex(class_order, fill_value=0)
        .reset_index()
    )
    class_count.columns = ["Class", "Count"]
    class_count["Class"] = class_count["Class"].astype(str)
    fig_class = px.bar(
        class_count,
        x="Class",
        y="Count",
        text="Count",
        color="Count",
        color_continuous_scale="Peach"
    )
    fig_class.update_layout(
        xaxis_title="클래스",
        yaxis_title="건수",
        showlegend=False,
        xaxis=dict(type='category'),
        shapes=[dict(
            type="rect", xref="paper", yref="paper",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(color="lightgray", width=1, dash="dash")
        )]
    )
    fig_class.update_coloraxes(showscale=False)
    st.plotly_chart(fig_class, use_container_width=True)

# ---------------------- 머신러닝 ----------------------    
elif selected == "머신러닝 시뮬레이터":
    st.markdown(
        """
        <h1 style="
            color: black; 
            font-size: 48px; 
            font-weight: 600; 
            text-align: center;
            ">
            머신러닝 결함 Class 예측 시뮬레이터
        </h1>
        """, 
        unsafe_allow_html=True
    )
    st.markdown("----------------------------------------------------")
    
    col1, col2 = st.columns(2)

    with col1:
        st.header("결함 특성값 입력")
        user_input = {}
        for feature in feature_m:
            vmin, vmax = feature_ranges[feature]
            default_val = (vmin + vmax) / 2
            user_input[feature] = st.slider(
                label=f"{feature} (범위: {vmin} ~ {vmax})",
                min_value=float(vmin),
                max_value=float(vmax),
                value=float(default_val),
                step=(float(vmax) - float(vmin)) / 1000
            )

    with col2:
        st.header("예측 결과")
        if st.button("결함 Class 예측하기"):
            input_df = pd.DataFrame([user_input])
            
            prob_results = {}
            for defect, model in models.items():
                proba_all = model.predict_proba(input_df)[0]
                
                # positive 클래스 확률이 1번 인덱스가 맞는지 확인 필요
                proba = proba_all[1] if len(proba_all) > 1 else proba_all[0]
                prob_results[defect] = proba
            
            prob_df = pd.DataFrame.from_dict(prob_results, orient='index', columns=["Probability"])
            prob_df.index = prob_df.index.astype(int)
            
            desired_order = [10, 14, 17, 20, 21, 22, 28, 39, 56, 99]
            prob_df = prob_df.reindex(desired_order, fill_value=0)
            
            top_class = prob_df["Probability"].idxmax()
            top_prob = prob_df.loc[top_class, "Probability"]
            colors = ['salmon' if cls == top_class else 'lightgray' for cls in prob_df.index]
            
            st.markdown(
                f"""
                <div style='background-color:#fff5f5; padding:20px; border-radius:10px;'>
                    <h3 style='margin:0;'>예측: <span style="color:salmon;">Class {top_class}</span></h3>
                    <h3 style='margin:0;'>확률: <span style="color:salmon;">{top_prob*100:.2f}%</span></h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            fig = go.Figure(go.Bar(
                x=prob_df.index.astype(str),
                y=prob_df["Probability"],
                marker_color=colors,
                text=[f"{p*100:.2f}%" for p in prob_df["Probability"]],
                textposition='auto'
            ))
            fig.update_layout(
                title="결함 Class별 예측 확률",
                yaxis=dict(autorange=True),
                xaxis=dict(type='category', categoryorder='array', categoryarray=[str(i) for i in desired_order]),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
