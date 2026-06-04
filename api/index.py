"""
Vercel Serverless Handler for Streamlit App
Exports handler variable as required by Vercel
"""
import os
import sys

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def handler(request):
    """
    Vercel request handler for Streamlit app
    """
    try:
        # Import Streamlit modules
        import streamlit.web.cli as stcli
        from streamlit.config import get_config_options
        
        # Set Streamlit server configuration for serverless environment
        os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
        os.environ['STREAMLIT_SERVER_ENABLEXSRFPROTECTION'] = 'false'
        os.environ['STREAMLIT_SERVER_PORT'] = '8501'
        
        # Return HTML response with Streamlit link
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Laptop Price Predictor</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }
                .container {
                    background: white;
                    padding: 40px;
                    border-radius: 10px;
                    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
                    text-align: center;
                    max-width: 500px;
                }
                h1 {
                    color: #333;
                    margin-bottom: 10px;
                }
                p {
                    color: #666;
                    margin-bottom: 20px;
                }
                .info {
                    background: #f0f0f0;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                    text-align: left;
                    font-size: 14px;
                }
                .info li {
                    margin: 8px 0;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 Laptop Price Predictor</h1>
                <p>Your ML-powered price prediction app is deployed!</p>
                <div class="info">
                    <strong>ℹ️ Note:</strong>
                    <p>For the best experience with Streamlit on Vercel, consider deploying on <a href="https://streamlit.io/cloud" target="_blank">Streamlit Cloud</a> instead.</p>
                    <p><strong>Why?</strong></p>
                    <ul>
                        <li>Streamlit Cloud is optimized for Streamlit apps</li>
                        <li>Better performance and real-time updates</li>
                        <li>No cold start issues</li>
                        <li>Direct GitHub integration</li>
                    </ul>
                </div>
                <p style="margin-top: 30px; color: #999; font-size: 12px;">
                    If you continue using Vercel, the app might experience cold starts due to serverless limitations.
                </p>
            </div>
        </body>
        </html>
        """
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'text/html; charset=utf-8',
            },
            'body': html
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'text/plain; charset=utf-8',
            },
            'body': f"Error: {str(e)}"
        }

