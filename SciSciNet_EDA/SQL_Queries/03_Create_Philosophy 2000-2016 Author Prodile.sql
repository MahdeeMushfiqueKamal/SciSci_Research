CREATE OR REPLACE TABLE `sciscinet-mahdee.SciSciNet.PhilosophyAuthors_2000_16_Profile` AS
SELECT 
  a.AuthorID,
  AVG(Reference_Count) AS Avg_Reference_Count,
  AVG(Citation_Count) AS Avg_Citation_Count,
  AVG(C5) AS Avg_C5,
  AVG(LOG10(C5+1)) AS Avg_Log10_C5,
  MAX(C5) AS Max_C5,
  AVG(Disruption) AS Avg_Disruption,
  AVG(WSB_Cinf) AS Avg_WSB_Cinf,
  AVG(WSB_mu) AS Avg_WSB_mu,
  AVG(WSB_sigma) AS Avg_WSB_sigma,
  AVG(SB_B) AS Avg_SB_B,
  AVG(SB_T) AS Avg_SB_T,
  SUM(NSF_Count) AS Total_NSF_Count
FROM `sciscinet-mahdee.SciSciNet.PhilosophyAuthors_2000_16` a
INNER JOIN `sciscinet-mahdee.SciSciNet.SciSciNet_PaperAuthorAffiliations` pa
  ON a.AuthorID = pa.AuthorID 
INNER JOIN `sciscinet-mahdee.SciSciNet.SciSciNet_Papers` p 
  ON p.PaperID = pa.PaperID
WHERE p.Year < 2016
GROUP BY a.AuthorID