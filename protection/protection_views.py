import streamlit as st
import io
from PIL import Image
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
from protection_visuals import generate_heatmap, generate_histograms  


def protection_comparison_view(original_img, protected_img, metrics):
    """Render side-by-side comparison view with metrics"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Original Style")
        st.image(original_img, use_container_width=True)
    
    with col2:
        st.subheader("Protected Style")
        st.image(protected_img, use_container_width=True)
        
        # Download button
        buf = io.BytesIO()
        protected_img.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Protected",
            data=buf.getvalue(),
            file_name="protected_style.png",
            mime="image/png"
        )
    
    with col3:
        st.subheader("Difference Heatmap")
        st.image(generate_heatmap(original_img, protected_img), 
                use_container_width=True)
    
    with st.expander("📊 Protection Metrics", expanded=True):
        tab1, tab2, tab3 = st.tabs(["Similarity", "Image Quality", "Histograms"])
        
        with tab1:
            col1, col2 = st.columns(2)
            col1.metric("SSIM", f"{metrics['ssim']:.3f}", delta_color="off")
            col2.metric("PSNR", f"{metrics['psnr']:.2f} dB", delta_color="off")
        
        with tab2:
            col1, col2, col3 = st.columns(3)
            col1.metric("Max Change", f"{metrics['max_diff']:.1f} px")
            col2.metric("Mean Change", f"{metrics['mean_diff']:.2f} px")
            col3.metric("95th Percentile", f"{metrics['perc_95']:.2f} px")
        
        with tab3:
            fig = generate_histograms(original_img, protected_img)
            st.pyplot(fig)
            plt.close(fig)