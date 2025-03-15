-- AuthorIDs of all the authors who published a Philosophy paper within 2000-2016
CREATE OR REPLACE TABLE `sciscinet-mahdee.Philosophy.PhilosophyAuthors_2000_16` AS
SELECT DISTINCT AuthorID
FROM (
    SELECT a.AuthorID
    FROM `sciscinet-mahdee.SciSciNet.SciSciNet_Authors` a
    INNER JOIN `sciscinet-mahdee.SciSciNet.SciSciNet_PaperAuthorAffiliations` ap
        ON a.AuthorID = ap.AuthorID
    INNER JOIN `sciscinet-mahdee.Philosophy.PhilosophyPapers_2000_15` p  
        ON ap.PaperID = p.PaperID
    
    UNION ALL
    
    SELECT a.AuthorID
    FROM `sciscinet-mahdee.SciSciNet.SciSciNet_Authors` a
    INNER JOIN `sciscinet-mahdee.SciSciNet.SciSciNet_PaperAuthorAffiliations` ap
        ON a.AuthorID = ap.AuthorID
    INNER JOIN `sciscinet-mahdee.Philosophy.PhilosophyPapers_2016` p  
        ON ap.PaperID = p.PaperID
) combined_authors