# Bibliography Audit

Date: 2026-07-30

Scope: read-only audit of the active manuscript at
`/Users/wei/Code/Interface/active/overleaf/interface-reconstruction-paper`.
No manuscript or bibliography files were changed.

## Executive Summary

- **Blocking before submission:** replace four Semantic Scholar API placeholders with complete publisher metadata.
- **Blocking before submission:** remove the blanket `\nocite` lists or turn the intended entries into real citations. They currently force 22 references into the bibliography that are never cited in the paper.
- **Should fix:** change `Pilliod1992` from the unsupported BibTeX entry type `@thesis` to `@mastersthesis`.
- **Should fix:** normalize three DOI fields that incorrectly contain full `https://doi.org/` URLs.
- The current build has **no undefined citations**. It uses 43 of the 44 database entries because `\nocite` forces most of the database into the bibliography.

## Priority 0: Placeholder And Incomplete Entries

All four entries below use `https://api.semanticscholar.org/CorpusID:...` as their `url`. Replace those URLs and complete the metadata using the values below.

| Key | Current lines | Required metadata |
|---|---:|---|
| `Chen2024ACI` | `mybibfile.bib:1-7` | `journal={Physics of Fluids}`, `volume={36}`, `number={3}`, `pages={032128}`, `year={2024}`, `doi={10.1063/5.0200627}`, `url={https://doi.org/10.1063/5.0200627}` |
| `Remmerswaal2021ParabolicIR` | `mybibfile.bib:9-17` | Change publication year to `2022`; retain `volume={469}` and `pages={111473}`; add `doi={10.1016/j.jcp.2022.111473}` and DOI/publisher URL. The key may remain unchanged to avoid citation churn. |
| `Diwakar2009AQS` | `mybibfile.bib:19-27` | Add `number={24}` and `doi={10.1016/j.jcp.2009.09.014}`; normalize pages to `9107--9130`; replace the Semantic Scholar URL with the DOI/publisher URL. |
| `Maity2024PiecewiseCI` | `mybibfile.bib:29-37` | Add `number={4}` and `doi={10.1002/fld.5256}`; normalize pages to `574--599`; replace the Semantic Scholar URL with the DOI/publisher URL. |

Metadata sources:

- Chen et al.: <https://doi.org/10.1063/5.0200627>
- Remmerswaal and Veldman: <https://doi.org/10.1016/j.jcp.2022.111473>
- Diwakar et al.: <https://doi.org/10.1016/j.jcp.2009.09.014>
- Maity et al.: <https://onlinelibrary.wiley.com/doi/10.1002/fld.5256>

## Priority 1: DOI And Entry-Type Corrections

1. `mybibfile.bib:41`, `:471`, and `:485`: DOI fields must contain bare DOI identifiers, not full resolver URLs.
   - `10.1016/j.jcp.2023.112656`
   - `10.1016/j.jcp.2007.12.029`
   - `10.1016/j.jcp.2023.111998`
2. `mybibfile.bib:454`: change `@thesis{Pilliod1992` to `@mastersthesis{Pilliod1992`. The current Elsevier BibTeX style reports: `entry type for "Pilliod1992" isn't style-file defined`.
3. `mybibfile.bib:70`: verify and normally change `Maity2020`'s journal year from `2020` to `2021` (volume 93, issue 1); the DOI was published online in 2020, but the issue is dated 2021.
4. `mybibfile.bib:403`: correct author `Garimalla` to `Garimella` in `Kucharik2010`.
5. `mybibfile.bib:162`: correct `An interfacing tracking method` to `An interface tracking method` in `Youngs1987`.
6. `mybibfile.bib:97-106`: add Hirt and Nichols' DOI `10.1016/0021-9991(81)90145-5` and remove the incorrect `publisher={Citeseer}` field.

Optional cleanup after the blockers:

- Normalize page ranges to BibTeX `--` throughout.
- Remove exported `bdsk-url-*`, `abstract`, and `keywords` fields that are not used by the journal style.
- Normalize journal names consistently, either full names or the journal's accepted abbreviations.

## Citation Reachability

Database entries: **44**

Actually cited in manuscript prose: **21**

Included only because of `\nocite`: **22**

Completely unused: **1** (`Huettenberger2013`)

Undefined citation keys: **0**

The `\nocite` directives are at `interface-reconstruction.tex:100-103`. Remove them before submission. If any of the 22 works are important to the scientific positioning, cite them at the relevant claim instead of retaining an unstructured reference dump.

Entries currently included only through `\nocite`:

`Aulisa2007`, `Benson1992`, `Bonnell2003`, `Bornia2011`, `Dyadechko2005`, `Francois2010`, `Grandy1999`, `Kucharik2010`, `Lopez2004`, `Meredith2005`, `Peery2000`, `Popinet1999`, `Renardy2002`, `Scheffler2000`, `Schofield2009`, `Sethian2003`, `Tryggvason2001`, `Unverdi1992`, `Weiler1977`, `Winslow1981`, `Zaleski2003`, `Zhang2014`.

## Verification

The active source was compiled from the Overleaf parent directory with bibliography and style search paths pointed at the paper folder. Result:

- PDF: 47 pages
- Undefined citations: 0
- Undefined references: 0
- BibTeX warnings: 1 (`Pilliod1992` entry type)
- Minor layout warnings remain, documented in the submission-package audit.
