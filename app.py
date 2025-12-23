import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


st.set_page_config(
    page_title="MarketMind: AI Customer Segmentation",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .block-container {padding-top: 1rem;}
    h1 {color: #0E1117;}
    .stButton>button {width: 100%; border-radius: 5px; height: 3em;}
    </style>
    """, unsafe_allow_html=True)


def get_persona(income, score):
    """
    Translates mathematical coordinates into business personas.
    Note: These thresholds assume the standard blob generation range (-10 to 10 approx).
    """
    if income > 4 and score > 4:
        return "💎 VIP / Big Spender", "Target with luxury products and exclusive offers."
    elif income > 4 and score < -2:
        return "💰 Frugal / Saver", "Target with 'Best Value' deals and bulk discounts."
    elif income < -2 and score > 4:
        return "💸 Impulse Buyer", "Target with limited-time offers and payment plans."
    elif income < -2 and score < -2:
        return "📉 Budget Conscious", "Target with clearance sales and essential items."
    else:
        return "⚖️ Average Customer", "Target with standard loyalty programs."

@st.cache_data
def load_data(n_samples, random_state):
    
    X, _ = make_blobs(n_samples=n_samples, centers=5, cluster_std=0.9, random_state=random_state)
  
    df = pd.DataFrame(X, columns=['Annual_Income', 'Spending_Score'])
    return df


with st.sidebar:
    st.title("⚙️ Control Panel")
    
    st.subheader("Data Generation")
    n_samples = st.slider("Number of Customers", 200, 2000, 500)
    rand_state = st.number_input("Random Seed (Change Data)", value=42, step=1)
    
    st.markdown("---")
    st.subheader("AI Model Settings")
    n_clusters = st.slider("Select K (Clusters)", 2, 10, 5)
    
    st.markdown("---")
    st.info("💡 **Tip:** Use the 'Elbow Method' tab to verify if your chosen K is mathematically optimal.")

st.title("🛒 MarketMind: AI Customer Segmentation")
st.markdown("An unsupervised machine learning tool that groups customers into **Targetable Segments**.")


df = load_data(n_samples, rand_state)


kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Annual_Income', 'Spending_Score']])
df['Cluster'] = df['Cluster'].astype(str) # Convert to string for categorical coloring


tab1, tab2, tab3 = st.tabs(["📊 Dashboard & Analysis", "🔮 Predict New Customer", "📥 Raw Data"])


with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Interactive Customer Segments")
       
        fig = px.scatter(
            df, 
            x='Annual_Income', 
            y='Spending_Score', 
            color='Cluster',
            title=f"Customer Segments (K={n_clusters})",
            template="plotly_white",
            hover_data=['Annual_Income', 'Spending_Score'],
            color_discrete_sequence=px.colors.qualitative.Bold
        )
       
        centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['Annual_Income', 'Spending_Score'])
        fig.add_scatter(
            x=centroids['Annual_Income'], 
            y=centroids['Spending_Score'], 
            mode='markers', 
            marker=dict(color='black', size=15, symbol='star'),
            name='Centroids'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("The Elbow Method")
        st.write("Find the 'bend' to pick the best K.")
        
       
        wcss = []
        for i in range(1, 11):
            km = KMeans(n_clusters=i, init='k-means++', random_state=42)
            km.fit(df[['Annual_Income', 'Spending_Score']])
            wcss.append(km.inertia_)
            
        fig_elbow = px.line(
            x=range(1, 11), 
            y=wcss, 
            markers=True,
            labels={'x': 'Number of Clusters', 'y': 'WCSS (Error)'},
            template="plotly_white"
        )
        fig_elbow.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_elbow, use_container_width=True)


with tab2:
    st.subheader("🤖 Artificial Intelligence Consultant")
    st.write("Enter a new customer's details to classify them instantly.")
    
    c1, c2, c3 = st.columns([1, 1, 2])
    
    with c1:
        income_in = st.number_input("Annual Income (Normalized)", -12.0, 12.0, 5.0)
    with c2:
        score_in = st.number_input("Spending Score (Normalized)", -12.0, 12.0, 5.0)
    
    if st.button("Analyze Persona"):
       
        input_data = np.array([[income_in, score_in]])
        pred_cluster = kmeans.predict(input_data)[0]
        
      
        persona, advice = get_persona(income_in, score_in)
        
       
        st.divider()
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric(label="Assigned Cluster", value=f"Group {pred_cluster}")
            st.markdown(f"### {persona}")
            
        with res_col2:
            st.info(f"**Marketing Strategy:**\n\n{advice}")
            
       
        st.write("position relative to others:")
       
        user_point = pd.DataFrame({'Annual_Income': [income_in], 'Spending_Score': [score_in]})
        
        mini_fig = px.scatter(df, x='Annual_Income', y='Spending_Score', color_discrete_sequence=['lightgray'])
        mini_fig.add_scatter(x=user_point['Annual_Income'], y=user_point['Spending_Score'], 
                             mode='markers', marker=dict(color='red', size=20, symbol='diamond'), name='New Customer')
        mini_fig.update_layout(height=250, showlegend=False, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(mini_fig, use_container_width=True)


with tab3:
    st.subheader("📥 Data Export")
    st.write("Download the segmented data for your marketing team.")
    
    col_d1, col_d2 = st.columns([2, 1])
    
    with col_d1:
        st.dataframe(df, use_container_width=True)
        
    with col_d2:
        st.write("### Summary Stats")
        st.write(df.groupby('Cluster').mean())
        
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download CSV Report",
            data=csv,
            file_name='segmented_customers.csv',
            mime='text/csv',
        )
