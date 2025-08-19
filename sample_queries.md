# Sample Queries & Expected Behaviors

1. **Query:** "Data scientist with 3+ years experience in NLP, preferably in Bangalore."  
   **Expected Behavior:**  
   - The system should prioritize candidates who have experience in NLP and at least 3 years of relevant work.  
   - Candidates based in Bangalore will get a slight ranking boost.  
   - Each top candidate’s explanation should clearly show which skills matched, their location, and years of experience.

2. **Query:** "Frontend engineer skilled in React and TypeScript, 2+ years experience."  
   **Expected Behavior:**  
   - Candidates with strong React and TypeScript skills and at least 2 years of experience should rank higher.  
   - Other relevant frontend skills, like JavaScript or CSS, can help improve their score if contextually relevant.  
   - The explanation should list matched skills versus missing skills.

3. **Query:** "Machine Learning engineer with 2+ years experience in deep learning and Python."  
   **Expected Behavior:**  
   - Candidates with experience in ML, deep learning, and Python should be ranked highest.  
   - Only candidates with a minimum of 2 years’ experience are considered.  
   - Explanations should highlight matched skills and provide a similarity score showing how well the candidate matches the query.

4. **Query:** "Full-stack developer with Node.js, MongoDB, React experience, 3+ years, in Delhi."  
   **Expected Behavior:**  
   - Candidates who are full-stack developers with the listed skills and at least 3 years of experience will appear at the top.  
   - Candidates located in Delhi get extra consideration in the ranking.  
   - Each explanation shows which skills matched, which are missing, and their experience and location.

5. **Query:** "AI engineer with experience in recommendation systems, Spark, and Python, 3+ years."  
   **Expected Behavior:**  
   - The system should favor candidates with AI and recommendation system experience, as well as Spark and Python skills.  
   - Only candidates with at least 3 years of experience are considered.  
   - Explanations should clearly outline matched skills, missing critical skills, years of experience, location, and similarity score.l