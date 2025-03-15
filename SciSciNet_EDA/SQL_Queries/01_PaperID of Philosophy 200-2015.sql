CREATE OR REPLACE TABLE `sciscinet-mahdee.SciSciNet.PhilosophyPapers_2000_15` AS
SELECT p.PaperID 
FROM `sciscinet-mahdee.SciSciNet.SciSciNet_Papers` p  
INNER JOIN `sciscinet-mahdee.SciSciNet.SciSciNet_PaperFields` pf 
    ON p.PaperID = pf.PaperID 
INNER JOIN `sciscinet-mahdee.SciSciNet.SciSciNet_Fields` f 
    ON pf.FieldID = f.FieldID 
WHERE 
  p.Year > 2000 AND
  p.Year <= 2015 AND
  f.Field_Name = "Philosophy";