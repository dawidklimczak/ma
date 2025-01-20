import streamlit as st
import requests
from bs4 import BeautifulSoup
import html2image
from PIL import Image
import io
import numpy as np
from collections import Counter
import re
import pandas as pd
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def fetch_html_content(url):
    """Pobiera zawartość HTML z podanego URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        st.error(f"Błąd podczas pobierania zawartości z {url}: {str(e)}")
        return None

def analyze_text_content(html_content):
    """Analizuje treść tekstową."""
    soup = BeautifulSoup(html_content, 'html.parser')
    text = soup.get_text()
    
    # Liczba znaków
    char_count = len(text)
    
    # Najpopularniejsze słowa
    words = re.findall(r'\w+', text.lower())
    word_counts = Counter(words).most_common(10)
    
    return {
        'char_count': char_count,
        'popular_words': dict(word_counts)
    }

def html_to_image(html_content):
    """Konwertuje HTML do obrazu."""
    hti = html2image.Html2Image()
    try:
        img_bytes = hti.screenshot(html_str=html_content, save_as='temp.png')
        return Image.open('temp.png')
    except Exception as e:
        st.error(f"Błąd podczas konwersji HTML do obrazu: {str(e)}")
        return None

def analyze_image(image):
    """Analizuje obraz pod kątem kolorów i proporcji."""
    # Konwersja do tablicy numpy
    img_array = np.array(image)
    
    # Analiza kolorów
    pixels = img_array.reshape(-1, 3)
    kmeans = KMeans(n_clusters=5, random_state=42).fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    
    # Dominujący kolor
    dominant_color = tuple(colors[0])
    
    # Liczba unikalnych kolorów (przybliżona)
    unique_colors = len(np.unique(pixels, axis=0))
    
    # Analiza proporcji tekstu do grafiki
    gray = image.convert('L')
    threshold = 200  # próg binaryzacji
    binary = gray.point(lambda x: 0 if x < threshold else 255, '1')
    text_pixels = np.sum(np.array(binary) == 0)
    total_pixels = binary.width * binary.height
    text_ratio = text_pixels / total_pixels
    
    return {
        'dominant_color': dominant_color,
        'unique_colors': unique_colors,
        'text_ratio': text_ratio,
        'height_px': image.height
    }

def main():
    st.title('Analizator Kreacji Mailowych')
    
    # Pole do wprowadzania URL-i
    urls_input = st.text_area(
        "Wprowadź adresy URL kreacji (każdy w nowej linii)",
        height=150
    )
    
    if st.button('Analizuj'):
        urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
        
        if not urls:
            st.warning('Wprowadź co najmniej jeden adres URL')
            return
        
        results = []
        
        with st.spinner('Analizuję kreacje...'):
            for url in urls:
                st.write(f"Analizuję: {url}")
                
                # Pobieranie zawartości
                html_content = fetch_html_content(url)
                if not html_content:
                    continue
                
                # Analiza tekstu
                text_analysis = analyze_text_content(html_content)
                
                # Konwersja do obrazu i analiza
                image = html_to_image(html_content)
                if not image:
                    continue
                
                image_analysis = analyze_image(image)
                
                # Łączenie wyników
                result = {
                    'url': url,
                    'char_count': text_analysis['char_count'],
                    'popular_words': text_analysis['popular_words'],
                    'height_px': image_analysis['height_px'],
                    'dominant_color': image_analysis['dominant_color'],
                    'unique_colors': image_analysis['unique_colors'],
                    'text_ratio': image_analysis['text_ratio']
                }
                
                results.append(result)
        
        if results:
            # Wyświetlanie wyników
            st.subheader('Wyniki analizy')
            for result in results:
                st.write('---')
                st.write(f"URL: {result['url']}")
                st.write(f"Liczba znaków: {result['char_count']}")
                st.write(f"Wysokość (px): {result['height_px']}")
                st.write(f"Dominujący kolor (RGB): {result['dominant_color']}")
                st.write(f"Liczba unikalnych kolorów: {result['unique_colors']}")
                st.write(f"Proporcja tekstu do grafiki: {result['text_ratio']:.2%}")
                
                st.write("Najpopularniejsze słowa:")
                st.json(result['popular_words'])
            
            # Eksport do CSV
            df = pd.DataFrame(results)
            df['popular_words'] = df['popular_words'].apply(str)  # konwersja słownika na string
            csv = df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="Pobierz wyniki jako CSV",
                data=csv,
                file_name="analiza_kreacji.csv",
                mime="text/csv"
            )

if __name__ == '__main__':
    main()