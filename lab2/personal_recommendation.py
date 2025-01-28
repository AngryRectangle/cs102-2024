import pandas as pd
from joblib import load

svd_model = load("svd.pkl")
linreg_model, scaler, vectorizer = load("sgd.pkl")

user_zeros = ratings[ratings["Book-Rating"] == 0]["User-ID"].value_counts()
target_user = user_zeros.idxmax()
target_books = ratings[(ratings["User-ID"] == target_user) & (ratings["Book-Rating"] == 0)].merge(books, on="ISBN", how="inner")
svd_predictions = [
    (row["ISBN"], row["Book-Title"], row["Book-Author"], svd_model.predict(target_user, row["ISBN"]).est)
    for _, row in target_books.iterrows()
]

recommended_books = [(isbn, title, author, rating) for isbn, title, author, rating in svd_predictions if rating >= 8]
if recommended_books:
    rec_books_data = target_books[target_books["ISBN"].isin([isbn for isbn, _, _, _ in recommended_books])]
    title_vectors = vectorizer.transform(rec_books_data["Book-Title"].fillna("")).toarray()

    for col in ['Book-Author', 'Publisher']:
        rec_books_data[col] = rec_books_data[col].fillna("unknown").map(
            rec_books_data[col].value_counts(normalize=True).to_dict()
        )

    X_numeric = rec_books_data[['Year-Of-Publication', 'Book-Author', 'Publisher']]
    X_scaled = scaler.transform(X_numeric)

    X = pd.concat([
        pd.DataFrame(title_vectors, columns=[f"title_{i}" for i in range(title_vectors.shape[1])]),
        pd.DataFrame(X_scaled, columns=X_numeric.columns)
    ], axis=1).to_numpy()

    linreg_predictions = linreg_model.predict(X)
    recommendations = sorted(
        zip(rec_books_data["ISBN"], rec_books_data["Book-Title"], rec_books_data["Book-Author"], linreg_predictions),
        key=lambda x: x[3],
        reverse=True
    )
else:
    recommendations = []

if recommendations:
    for isbn, title, author, predicted_rating in recommendations:
        print(f"ISBN: {isbn}, Title: {title}, Predicted Rating (LinReg): {predicted_rating:.2f}")
else:
    print("No books(")
