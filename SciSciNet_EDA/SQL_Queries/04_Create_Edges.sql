CREATE OR REPLACE TABLE `sciscinet-mahdee.Philosophy.Philosophy_2000_15_edges` AS
SELECT AuthorID, p.PaperID FROM 
`sciscinet-mahdee.Philosophy.PhilosophyPapers_2000_15` p 
INNER JOIN 
`sciscinet-mahdee.SciSciNet.SciSciNet_PaperAuthorAffiliations` pa   
ON p.PaperID = pa.PaperID 
