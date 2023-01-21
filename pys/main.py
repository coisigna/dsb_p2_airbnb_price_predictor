import streamlit as st
import streamlit.components.v1 as com

st.header("Introduce your Airbnb data in the Typeform")

com.html(
    
"""
<!DOCTYPE html>
  <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>My typeform</title>
      <style>*{margin:0;padding:0;} html,body,#wrapper{width:100%;height:100%;} iframe{border-radius:0 !important;}</style>
    </head>
    <body>
      <div data-tf-widget="czGZ7ozW" id="wrapper" data-tf-opacity="100" data-tf-inline-on-mobile data-tf-iframe-props="title=My typeform" data-tf-transitive-search-params data-tf-auto-focus></div><script src="//embed.typeform.com/next/embed.js"></script>
    </body>
  </html>

""", height= 1000 )