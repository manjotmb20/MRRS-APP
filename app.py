import streamlit as st
from recipe_app import search_and_score_recipes  # Import from your main script

def main():
    st.title("Recipe Search and Translation Tool")

    user_query = st.text_input("Enter your recipe query:")
    if st.button("Search"):
        if user_query:
            results = search_and_score_recipes(user_query)
            for lang, result in results.items():
                st.subheader(f"Results for {lang.upper()}")
                for idx, (recipe, score) in enumerate(result, 1):
                    st.write(f"{idx}. Recipe: {recipe['recipeName']}\n   Score: {score}\n")
        else:
            st.write("Please enter a query to search.")

if __name__ == "__main__":
    main()
