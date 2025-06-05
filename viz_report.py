import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import io
import base64
import numpy as np # Import numpy here if not already imported

# Function to generate plots and save them to a BytesIO object
def create_plot_image(df, plot_type, x_col=None, y_col=None, color_col=None):
    fig, ax = plt.subplots()
    try:
        if plot_type == "histogram" and x_col:
            sns.histplot(df[x_col], ax=ax, kde=True)
            ax.set_title(f"Histogram of {x_col}")
        elif plot_type == "scatterplot" and x_col and y_col:
            sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
            ax.set_title(f"Scatter Plot of {x_col} vs {y_col}")
        elif plot_type == "boxplot" and x_col and y_col:
            sns.boxplot(x=df[x_col], y=df[y_col], ax=ax)
            ax.set_title(f"Box Plot of {y_col} by {x_col}")
        elif plot_type == "heatmap":
            corr = df.select_dtypes(include=np.number).corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            ax.set_title("Correlation Heatmap")
        else:
            ax.text(0.5, 0.5, "Plot not generated or invalid selection.", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        plt.close(fig)
        return buf
    except Exception as e:
        st.error(f"Error generating plot for report: {e}")
        plt.close(fig) # Ensure figure is closed even on error
        return None


class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Visio AI - Data Analysis Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()

    def add_dataframe(self, df, title=""):
        self.ln(5)
        self.set_font('Arial', 'B', 10)
        self.cell(0, 10, title, 0, 1, 'L')
        self.set_font('Arial', '', 8)

        # Calculate column widths
        col_widths = []
        for col in df.columns:
            # Estimate width based on header and max content length
            max_len = max(len(str(df[col].max())), len(col))
            col_widths.append(self.get_string_width(col) + 6) # Add padding
        
        # Adjust widths to fit page if needed, or set a max width
        page_width = self.w - self.l_margin - self.r_margin
        total_width = sum(col_widths)
        if total_width > page_width:
            scale_factor = page_width / total_width
            col_widths = [w * scale_factor for w in col_widths]


        # Table Header
        for i, header in enumerate(df.columns):
            self.cell(col_widths[i], 7, str(header), 1, 0, 'C')
        self.ln()

        # Table Rows
        self.set_font('Arial', '', 7)
        for index, row in df.iterrows():
            for i, val in enumerate(row.values):
                self.cell(col_widths[i], 7, str(val), 1, 0, 'L')
            self.ln()
        self.ln(5)

    def add_matplotlib_figure(self, fig_buffer, width=150):
        if fig_buffer:
            self.image(name=fig_buffer, type='PNG', w=width)
            self.ln(10)

def viz_report_app():
    st.markdown("<h2 style='text-align: center; color: #4A90E2;'>📄 Viz Report Generator</h2>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    st.info("This feature generates a PDF report based on your uploaded dataset and analysis.")

    if st.session_state.updated_df is not None:
        df = st.session_state.updated_df
        
        st.subheader("Report Configuration")
        report_title = st.text_input("Enter report title", "Data Analysis Report")
        include_plots = st.checkbox("Include generated plots (Histograms, Scatter Plots, Heatmap)")
        
        # Allow selection of specific plots if 'include_plots' is checked
        selected_report_plots = []
        if include_plots:
            st.markdown("**Select plots to include in the report:**")
            numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df.select_dtypes(include='object').columns.tolist()

            if st.checkbox("Include Histograms"):
                selected_report_plots.append("histogram")
            if st.checkbox("Include Scatter Plots (select axes below)"):
                selected_report_plots.append("scatterplot")
                st.session_state.report_scatter_x = st.selectbox("Scatter Plot X-axis", numerical_cols, key='report_scatter_x')
                st.session_state.report_scatter_y = st.selectbox("Scatter Plot Y-axis", numerical_cols, key='report_scatter_y')
            if st.checkbox("Include Box Plots (select axes below)"):
                selected_report_plots.append("boxplot")
                st.session_state.report_boxplot_num = st.selectbox("Box Plot Numerical Column", numerical_cols, key='report_boxplot_num')
                st.session_state.report_boxplot_cat = st.selectbox("Box Plot Categorical Column", categorical_cols, key='report_boxplot_cat')
            if st.checkbox("Include Correlation Heatmap"):
                selected_report_plots.append("heatmap")


        if st.button("Generate PDF Report"):
            with st.spinner("Generating PDF report... This may take a moment."):
                pdf = PDF()
                pdf.add_page()
                pdf.set_title(report_title)

                # Title
                pdf.set_font('Arial', 'B', 16)
                pdf.cell(0, 10, report_title, 0, 1, 'C')
                pdf.ln(10)

                # Dataset Overview
                pdf.chapter_title("1. Dataset Overview")
                pdf.chapter_body(f"Report generated for dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
                
                # Display first few rows of the DataFrame
                pdf.add_dataframe(df.head(), "First 5 Rows of Dataset")

                # Statistical Summary
                pdf.chapter_title("2. Statistical Summary of Numerical Columns")
                numeric_df_desc = df.select_dtypes(include=np.number).describe().reset_index()
                pdf.add_dataframe(numeric_df_desc, "Numerical Column Statistics")

                # Missing Values
                pdf.chapter_title("3. Missing Values Report")
                null_counts = df.isnull().sum()
                null_report = null_counts[null_counts > 0].reset_index()
                null_report.columns = ['Column', 'Missing Count']
                if not null_report.empty:
                    pdf.add_dataframe(null_report, "Columns with Missing Values")
                else:
                    pdf.chapter_body("No missing values found in the dataset.")

                # Model Training Summary (if a model was trained)
                if st.session_state.trained_model is not None:
                    pdf.chapter_title("4. Machine Learning Model Summary")
                    pdf.chapter_body(f"Problem Type: {st.session_state.problem_type.capitalize()}")
                    pdf.chapter_body(f"Algorithm Used: {st.session_state.selected_algo_name}")
                    pdf.chapter_body(f"Target Column: {st.session_state.target_column}")
                    pdf.chapter_body(f"Features Used: {', '.join(st.session_state.feature_columns)}")

                    if st.session_state.model_metrics:
                        pdf.chapter_body("Model Performance Metrics:")
                        metrics_str = ""
                        for metric, value in st.session_state.model_metrics.items():
                            if isinstance(value, np.ndarray): # For confusion matrix
                                metrics_str += f"{metric}:\n{str(value)}\n"
                            else:
                                metrics_str += f"{metric}: {value:.4f}\n"
                        pdf.chapter_body(metrics_str)

                # Include Plots
                if include_plots and selected_report_plots:
                    pdf.chapter_title("5. Visualizations")
                    for plot_name in selected_report_plots:
                        if plot_name == "histogram":
                            for col in df.select_dtypes(include=np.number).columns:
                                plot_buf = create_plot_image(df, "histogram", x_col=col)
                                pdf.add_matplotlib_figure(plot_buf)
                                pdf.chapter_body(f"Histogram of {col}")
                        elif plot_name == "scatterplot":
                            if 'report_scatter_x' in st.session_state and 'report_scatter_y' in st.session_state:
                                plot_buf = create_plot_image(df, "scatterplot", x_col=st.session_state.report_scatter_x, y_col=st.session_state.report_scatter_y)
                                pdf.add_matplotlib_figure(plot_buf)
                                pdf.chapter_body(f"Scatter Plot of {st.session_state.report_scatter_x} vs {st.session_state.report_scatter_y}")
                        elif plot_name == "boxplot":
                            if 'report_boxplot_num' in st.session_state and 'report_boxplot_cat' in st.session_state:
                                plot_buf = create_plot_image(df, "boxplot", x_col=st.session_state.report_boxplot_cat, y_col=st.session_state.report_boxplot_num)
                                pdf.add_matplotlib_figure(plot_buf)
                                pdf.chapter_body(f"Box Plot of {st.session_state.report_boxplot_num} by {st.session_state.report_boxplot_cat}")
                        elif plot_name == "heatmap":
                            plot_buf = create_plot_image(df, "heatmap")
                            pdf.add_matplotlib_figure(plot_buf)
                            pdf.chapter_body("Correlation Heatmap")
                        
            # Output PDF
            pdf_output = pdf.output(dest='S').encode('latin1')
            b64_pdf = base64.b64encode(pdf_output).decode('latin1')

            # Create download link
            href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="VisioAI_Report.pdf">Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.success("PDF report generated successfully!")

    else:
        st.warning("Please upload a dataset on the Home page to generate a Viz Report.")

# To run this as a standalone page when called by the main app
def app():
    viz_report_app()

if __name__ == "__main__":
    st.set_page_config("Viz Report Generator", layout='wide')
    viz_report_app()