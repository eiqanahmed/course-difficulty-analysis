import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# Load summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load and clean data
df = pd.read_csv("../data_scraping/cleaned_reviews.csv")
df = df.dropna(subset=["comment", "quality", "difficulty"])
df["thumbs_up"] = df["thumbs_up"].fillna(0)
df["thumbs_down"] = df["thumbs_down"].fillna(0)
df["grade"] = df["grade"].fillna("Unknown")

# Store results
examples = []
grouped = df.groupby(["normalized_course_code", "prof_id"])

for (course, prof_id), group in tqdm(grouped, desc="Summarizing aggregated comments"):
    comments = list(dict.fromkeys(group["comment"].dropna().astype(str).tolist()))
    full_text = " ".join(comments)

    avg_quality = round(group["quality"].mean(), 2)
    avg_difficulty = round(group["difficulty"].mean(), 2)
    thumbs_up = int(group["thumbs_up"].sum())
    thumbs_down = int(group["thumbs_down"].sum())
    grade_mode = group["grade"].mode()[0] if not group["grade"].mode().empty else "Unknown"

    metadata = (
        f"Course: {course}\n"
        f"Professor ID: {int(prof_id)}\n"
        f"Avg Quality: {avg_quality}\n"
        f"Avg Difficulty: {avg_difficulty}\n"
        f"Thumbs Up: {thumbs_up}\n"
        f"Thumbs Down: {thumbs_down}\n"
        f"Grade Mode: {grade_mode}\n"
    )

    # Token-safe truncation
    max_total_length = 1500
    max_comment_length = max_total_length - len(metadata)
    trimmed_text = full_text[:max_comment_length]
    input_text = metadata + trimmed_text

    try:
        summary = summarizer(
            input_text,
            max_length=80,
            min_length=40,
            do_sample=False,
            length_penalty=2.0
        )[0]["summary_text"]
    except Exception as e:
        summary = "Error generating summary"

    examples.append({
        "course_code": course,
        "prof_id": prof_id,
        "summary": summary.strip(),
        "metadata": metadata.strip()
    })

# Save as DataFrame
summary_df = pd.DataFrame(examples)
summary_df.to_csv("summarized_reviews_full.csv", index=False)
