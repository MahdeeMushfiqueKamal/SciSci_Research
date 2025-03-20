## Dataset Curation

Philosophy Papers of 2016:

```sql
CREATE OR REPLACE TABLE `sciscinet-mahdee.Philosophy.Philosophy_2016_Papers` AS
SELECT p.PaperID, p.C5, log(p.C5 + 1) AS C5_log 
FROM `sciscinet-mahdee.SciSciNet.SciSciNet_Papers` p  
INNER JOIN `sciscinet-mahdee.SciSciNet.SciSciNet_PaperFields` pf 
    ON p.PaperID = pf.PaperID 
INNER JOIN `sciscinet-mahdee.SciSciNet.SciSciNet_Fields` f 
    ON pf.FieldID = f.FieldID 
WHERE 
  p.Year = 2016 AND
  f.Field_Name = "Philosophy";
```

Philosophy Authors of 2016:

```sql
CREATE OR REPLACE TABLE `sciscinet-mahdee.Philosophy.Philosophy_2016_Authors_temp` AS
SELECT distinct a.AuthorID
FROM `sciscinet-mahdee.Philosophy.Philosophy_2016_Papers` p
INNER JOIN `sciscinet-mahdee.SciSciNet.SciSciNet_PaperAuthorAffiliations` pa 
ON p.PaperID = pa.PaperID
INNER JOIN `sciscinet-mahdee.SciSciNet.SciSciNet_Authors` a
ON a.AuthorID = pa.AuthorID;
```

Creating a User Profile Of Each AUthor Based on their papers before 2016: 

```sql
CREATE OR REPLACE TABLE `sciscinet-mahdee.Philosophy.Philosophy_2016_Author_Profiles` AS
SELECT 
  a.AuthorID,
  AVG(Reference_Count) AS Avg_Reference_Count,
  AVG(Citation_Count) AS Avg_Citation_Count,
  COUNT(*) AS Num_Of_Paper_Before_2015,
  AVG(C5) AS Avg_C5,
  AVG(LOG10(C5+1)) AS Avg_Log10_C5,
  MAX(C5) AS Max_C5,
  AVG(Disruption) AS Avg_Disruption,
  AVG(WSB_Cinf) AS Avg_WSB_Cinf,
  AVG(WSB_mu) AS Avg_WSB_mu,
  AVG(WSB_sigma) AS Avg_WSB_sigma,
  AVG(SB_B) AS Avg_SB_B,
  AVG(SB_T) AS Avg_SB_T,
  SUM(NSF_Count) AS Total_NSF_Count,
  MIN(p.Year) AS First_Publication_Year
FROM `sciscinet-mahdee.Philosophy.Philosophy_2016_Authors_temp` a 
LEFT JOIN `sciscinet-mahdee.SciSciNet.SciSciNet_PaperAuthorAffiliations` pa
  ON a.AuthorID = pa.AuthorID 
INNER JOIN `sciscinet-mahdee.SciSciNet.SciSciNet_Papers` p 
  ON p.PaperID = pa.PaperID
WHERE p.Year < 2016
GROUP BY a.AuthorID
```


