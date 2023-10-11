# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Load libraries
#
#
#

# %% jupyter={"outputs_hidden": false}
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import pyreadstat
import skimpy
from prettytable import PrettyTable
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# %% [markdown]
# # Load the data

# %% jupyter={"outputs_hidden": false}
# Import data from SPSS and clean columns
df_survey, metadata = pyreadstat.read_sav("data/2021_ACTUALIDADES_DATOS.sav")
df_survey = skimpy.clean_columns(df_survey)
df_survey.head()


# %%
skimpy.skim(df_survey)

# %% [markdown]
# # Data preparation and cleaning

# %% [markdown]
# ## Data preparation

# %%
# Remove some unnecesary variables
df_survey = df_survey.drop(
    columns=[
        "cs_10_br_2",
        "cs_11_c",
        "cs_15_a",
        "cs_15_b",
        "cs_16",
        "cs_18",
        "edad_3",
        "educa_3",
        "factor",
        "id",
        "cuestionario",
        "hi",
        "mi",
        "hf",
        "mf",
    ]
)

# %%
# Drop NA values which are code as 99, 98 or 999
df_survey.replace(
    {99: np.nan, 98: np.nan, 999: np.nan},
    inplace=True,
)


# Drop columns with more than 50% of NA values
df_tmp = df_survey.drop(
    columns=[
        "rp_3",
        "rp_4",
        "af_2_minutos",
        "af_2_horas",
        "af_4_minutos",
        "af_4_horas",
        "af_6_minutos",
        "af_6_horas",
        "af_7_minutos",
        "af_7_horas",
        "cs_2",
        "cs_4",
    ],
    axis=1,
)

for i in df_tmp.columns:
    df_survey[i] = np.where(df_survey[i] == 9, np.nan, df_survey[i])
    df_survey[i] = np.where(df_survey[i] == 8, np.nan, df_survey[i])

# %%
# Remove String variables are dropped because we only need the numeric ones
idx = df_survey.select_dtypes(object).columns
df_survey = df_survey.drop(idx, axis=1)


# %% [markdown]
# After a throughfully revision, we notices that many variables aren't related with the infection risk. They are more opinions postures about some topic. 
#
# We will exclude all the variables starting with "sp", "rv", "cv", "va" (except va6 because is related with the vaccination status) and "nf".
#
#
# %% [markdown]
# ## Creation of class of variable with 3 levels
#
# This ways we can order the variables between `Low` (0), `Medium` (1) and `High` (2) risk.

# %% jupyter={"outputs_hidden": false}
cat_012_ord = pd.CategoricalDtype(categories=[0, 1, 2], ordered=True)


# %%
# This variable will contain the final set of variables to be used in the analysis
clean_cols = []

# %% [markdown]
# # Data cleaning

# %% [markdown]
# The paper links the determinants of the health with each of the variables in the dataset.  For a better understanding of the data, we will rename the variables with the name of the determinant of the health that they are related with.
#
# ![Alt text](image.png)
# _Source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7005090/ _
#
# In this spirit the variable will have a prefix with the determinant of the health that they are related with:
#
# - `sd`: Structural determinant 
# - `id`: Intermediate determinant
#
# Then, it follow the type of determinant. So for the structural determinants:
#
# - `sd_edu`: Education
# - `sd_eco`: Economic
# - `sd_occ`: Occupation
# - `sd_inc`: Income
# - `sd_eth`: Ethnicity
# - `sd_gen`: Gender
# - `sd_cul`: Cultural
#
# And for the intermediate determinants:
#
# - `id_beh`: Behavioral
# - `id_bio`: Biological
# - `id_psy`: Psychosocial
# - `id_mat`: Material
#
# Finally, the variable will have a suffix with the determinant of the health that they are related with. 
#
#
#  Determinantes sociales
#   Determinantes estructurales
#  contexto socioeconomico = de_cs
#  gobernanza = de_go
#  politicas (macroeconomicas, salud, sociedad)  = de_po
#  normas y valores culturales = de_cu
#  eduacion = de_ed
#  ocupacion = de_oc
#  ingresos = de_in
#  genero = de_ge
#  etnia = de_et
#  Determinantes intermedios
#  factores materiales = di_fm
#  factores psicosociales = di_fp
#  factores conductuales = di_fc
#  factores biológicos = di_fb

# %% [markdown]
# ## Covid-19 variable

# %% jupyter={"outputs_hidden": false}
df_survey = (
    df_survey.assign(
        covid19=lambda df: np.select(
            [
                (df.rp_1 == 1) | (df.rp_1 == 2),
                (df.rp_1 == 3),
                (df.rp_1 == 8) | (df.rp_1 == 9),
            ],
            [1, 0, np.nan],
            default=np.nan,
        )
    )
    .assign(covid19=lambda df: df.covid19.astype("bool"))
    .drop(["rp_1"], axis=1)
)


# %% jupyter={"outputs_hidden": false}
clean_cols.append("covid19")
df_survey.covid19.value_counts()

# %% [markdown]
# ## Self-perception of contagion risk
#
# The variable `rp_3 is the self contagiuos perception. We will change the values to 0 (Low), 1 (Medium) and 2 (High).
#  

# %% jupyter={"outputs_hidden": false}
df_survey = (
    df_survey.assign(
        id_beh_percep_contag=lambda df: np.select(
            [
                df.rp_3.between(0, 3),
                df.rp_3.between(4, 6),
                df.rp_3.between(7, 10),
                df.rp_3 == 99,
            ],
            [0, 1, 2, np.nan],
            default=np.nan,
        )
    )
    .assign(id_beh_percep_contag=lambda df: df.id_beh_percep_contag.astype(cat_012_ord))
    .drop(["rp_3"], axis=1)
)


# %% jupyter={"outputs_hidden": false}
clean_cols.append("id_beh_percep_contag")
df_survey.id_beh_percep_contag.value_counts(dropna=False)

# %% [markdown]
# ## Self-perception of severity of the disease

# %% jupyter={"outputs_hidden": false}
# Covid-19: Autopercepción gravedad
df_survey = (
    df_survey.assign(
        id_beh_percep_severity=lambda df: np.select(
            [
                df.rp_4.between(0, 3),
                df.rp_4.between(4, 6),
                df.rp_4.between(7, 10),
                df.rp_4 == 99,
            ],
            [0, 1, 2, np.nan],
            default=0,
        )
    )
    .assign(
        id_beh_percep_severity=lambda df: df.id_beh_percep_severity.astype(cat_012_ord)
    )
    .drop(["rp_4"], axis=1)
)


# %% jupyter={"outputs_hidden": false}
clean_cols.append("id_beh_percep_severity")
df_survey.id_beh_percep_severity.value_counts(dropna=False)

# %% [markdown]
# ## Contagion inside the household (Bubble)

# %% jupyter={"outputs_hidden": false}
df_survey = (
    df_survey.assign(
        id_bio_bubble_contag=lambda df: np.select(
            [
                (df.rp_5 == 1) | (df.rp_5 == 9),
                (df.rp_5 == 2),
            ],
            [1, 0],
            default=0,
        )
    )
    .assign(id_bio_bubble_contag=lambda df: df.id_bio_bubble_contag.astype("bool"))
    .drop(["rp_5"], axis=1)
)


# %%
clean_cols.append("id_bio_bubble_contag")
df_survey.id_bio_bubble_contag.value_counts(dropna=False)

# %% [markdown]
# ## Contagion outside the household (Bubble)

# %% jupyter={"outputs_hidden": false}
df_survey = (
    df_survey.assign(
        id_bio_out_bubble_contag=lambda df: np.select(
            [
                (df.rp_6 == 1) | (df.rp_6 == 9),
                (df.rp_6 == 2),
            ],
            [1, 0],
            default=0,
        )
    )
    .assign(
        id_bio_out_bubble_contag=lambda df: df.id_bio_out_bubble_contag.astype("bool")
    )
    .drop(["rp_6"], axis=1)
)


# %%
clean_cols.append("id_bio_out_bubble_contag")
df_survey.id_bio_out_bubble_contag.value_counts(dropna=False)

# %% [markdown]
# ## Aware of deaths by Covid-19
# %%
df_survey = (
    df_survey.assign(
        id_bio_death_covid=lambda df: np.select(
            [
                (df.rp_7 == 1) | (df.rp_7 == 9),
                (df.rp_7 == 2),
            ],
            [1, 0],
            default=0,
        )
    )
    .assign(id_bio_death_covid=lambda df: df.id_bio_death_covid.astype("bool"))
    .drop(["rp_7"], axis=1)
)

# %%
clean_cols.append("id_bio_death_covid")
df_survey.id_bio_death_covid.value_counts(dropna=False)

# %% [markdown]
# ## Risk behavior

# %% jupyter={"outputs_hidden": false}
df_survey = (
    df_survey.assign(
        cr_1=lambda df: np.where(df.cr_1 == 9, np.nan, df.cr_1),
        cr_2=lambda df: np.where(df.cr_2 == 9, np.nan, df.cr_2),
    )
    .assign(
        cr_1=lambda df: np.select(
            [
                df.cr_1.between(1, 2),
                df.cr_1.between(3, 3),
                df.cr_1.between(4, 5),
                df.cr_1 == 9,
            ],
            [2, 1, 0, np.nan],
            default=np.nan,
        ),
        cr_2=lambda df: np.select(
            [
                df.cr_2.between(1, 2),
                df.cr_2.between(3, 3),
                df.cr_2.between(4, 5),
                df.cr_2 == 9,
            ],
            [2, 1, 0, np.nan],
            default=np.nan,
        ),
    )
    .assign(id_beh_risk_personal=lambda df: df[["cr_1", "cr_2"]].median(axis=1).round())
    .assign(
        id_beh_risk_personal=lambda df: df.id_beh_risk_personal.astype(
            pd.CategoricalDtype(ordered=True)
        )
    )
    .drop(["cr_1", "cr_2"], axis=1)
)


# %% jupyter={"outputs_hidden": false}
clean_cols.append("id_beh_risk_personal")
df_survey.id_beh_risk_personal.value_counts()

# %% [markdown]
# ## Covid risk behaviors towards other people

# %% jupyter={"outputs_hidden": false}
df_survey = (
    # change the value of 9 in cr_3 and cr_4 by np.nan
    df_survey.assign(
        cr_3=lambda df: np.where(df.cr_3 == 9, np.nan, df.cr_3),
        cr_4=lambda df: np.where(df.cr_4 == 9, np.nan, df.cr_4),
    )
    # change the values of cr_3 and cr_4 by 0, 1, 2
    .assign(
        cr_3=lambda df: np.select(
            [
                df.cr_3.between(1, 2),
                df.cr_3.between(3, 3),
                df.cr_3.between(4, 5),
                df.cr_3 == 9,
            ],
            [0, 1, 2, np.nan],
            default=np.nan,
        ),
        cr_4=lambda df: np.select(
            [
                df.cr_4.between(1, 2),
                df.cr_4.between(3, 3),
                df.cr_4.between(4, 5),
                df.cr_4 == 9,
            ],
            [0, 1, 2, np.nan],
            default=np.nan,
        ),
    )
    .assign(id_beh_risk_others=lambda df: df[["cr_3", "cr_4"]].median(axis=1).round())
    .assign(id_beh_risk_others=lambda df: df.id_beh_risk_others.astype(cat_012_ord))
    .drop(["cr_3", "cr_4"], axis=1)
)


# %% jupyter={"outputs_hidden": false}
clean_cols.append("id_beh_risk_others")
df_survey.id_beh_risk_others.value_counts()

# %% [markdown]
# ## Physical activity
#
# The rationale behind this variable is that people who do physical activity are more likely to have a better health and therefore, a better immune system.
#
# To create the variable we use the following assumitions: 
#
# - If people don't do any physical activity, they are in the `High` (2) risk group.
# - If people at least do moderate or light physical activity, they are in the `Medium` (1) risk group.
# - If people do intense physical activity, they are in the `Low` (0) risk group.

# %% jupyter={"outputs_hidden": false}
df_survey = (
    df_survey.assign(
        af_1=lambda df: df.af_1.astype("float"),
        af_3=lambda df: df.af_3.astype("float"),
        af_5=lambda df: df.af_5.astype("float"),
        af_7_horas=lambda df: df.af_7_horas.astype("float"),
        af_7_minutos=lambda df: df.af_7_minutos.astype("float"),
    )
    .assign(
        af_1=lambda df: df.af_1.fillna(0),
        af_3=lambda df: df.af_3.fillna(0),
        af_5=lambda df: df.af_5.fillna(0),
        af_7_horas=lambda df: df.af_7_horas.fillna(0),
        af_7_minutos=lambda df: df.af_7_minutos.fillna(0),
    )
    # Transform each variable such as:
    # af_1, af_3, af_5 is 1 if less than 3 and 0 otherwise.
    .assign(
        id_beh_physical_act=lambda df: np.select(
            [
                (df.af_1 == 9) & (df.af_3 == 9) & (df.af_5 == 9),
                (df.af_1 == 0) & (df.af_3 == 0) & (df.af_5 == 0),
                ((df.af_3 > 0) | (df.af_5 > 0)) & (df.af_1 == 0),
                (df.af_1 > 0),
            ],
            [np.nan, 2, 1, 0],
            default=np.nan,
        )
    )
    .assign(id_beh_physical_act=lambda df: df.id_beh_physical_act.astype(cat_012_ord))
)


# %% jupyter={"outputs_hidden": false}
clean_cols.append("id_beh_physical_act")
df_survey.id_beh_physical_act.value_counts(dropna=False)

# %% [markdown]
# ## Vaccination myths

# %% jupyter={"outputs_hidden": false}
# Vacunacion

# Get the list of columns starting with 'vm'
vm_columns = [col for col in df_survey.columns if col.startswith("vm")]

# Apply the assign and np.select operations to each 'vm' column
for col in vm_columns:
    df_survey[col] = np.select(
        [
            df_survey[col].between(1, 2),
            df_survey[col] == 3,
            df_survey[col].between(4, 5),
            df_survey[col] == 9,
        ],
        [0, 1, 2, np.nan],
    )

df_survey = (
    df_survey.assign(id_psy_vacc_myths=lambda df: df[vm_columns].median(axis=1).round())
    .assign(id_psy_vacc_myths=lambda df: df.id_psy_vacc_myths.astype(cat_012_ord))
    .drop(columns=vm_columns)
)

# %% jupyter={"outputs_hidden": false}
clean_cols.append("id_psy_vacc_myths")
df_survey.id_psy_vacc_myths.value_counts()
# %% [markdown]
# ## Vaccination status

# %% jupyter={"outputs_hidden": false}
df_survey = (
    df_survey.assign(
        id_bio_vacc_status=lambda df: np.select(
            [df.va_6.between(1, 2), df.va_6 == 3, df.va_6 == 4, df.va_6 == 9],
            [0, 1, 2, np.nan],
            default=np.nan,
        )
    )
    .assign(id_bio_vacc_status=lambda df: df.id_bio_vacc_status.astype(cat_012_ord))
    .drop(columns=["va_6"])
)


# %% jupyter={"outputs_hidden": false}
clean_cols.append("id_bio_vacc_status")
df_survey.id_bio_vacc_status.value_counts()

# %% [markdown]
# ## Anxiety disorders

# %% jupyter={"outputs_hidden": false}
df_survey = df_survey.assign(
    an_1=lambda df: np.select(
        [df.an_1.between(0, 1), df.an_1 == 2, df.an_1.between(3, 4), df.an_1 == 9],
        [0, 1, 2, np.nan],
        default=np.nan,
    ),
    an_2=lambda df: np.select(
        [df.an_2.between(0, 1), df.an_2 == 2, df.an_2.between(3, 4), df.an_2 == 9],
        [0, 1, 2, np.nan],
        default=np.nan,
    ),
    an_3=lambda df: np.select(
        [df.an_3.between(0, 1), df.an_3 == 2, df.an_3.between(3, 4), df.an_3 == 9],
        [0, 1, 2, np.nan],
        default=np.nan,
    ),
    an_4=lambda df: np.select(
        [df.an_4.between(0, 1), df.an_4 == 2, df.an_4.between(3, 4), df.an_4 == 9],
        [0, 1, 2, np.nan],
        default=np.nan,
    ),
    an_5=lambda df: np.select(
        [df.an_5.between(0, 1), df.an_5 == 2, df.an_5.between(3, 4), df.an_5 == 9],
        [0, 1, 2, np.nan],
        default=np.nan,
    ),
    an_6=lambda df: np.select(
        [df.an_6 == 1, df.an_6 == 2, df.an_6 == 3, df.an_6 == 9],
        [0, 1, 2, np.nan],
        default=np.nan,
    ),
)

df_survey = (
    df_survey.assign(
        id_psy_anxiety_sympt=lambda df: df[
            [
                "an_1",
                "an_2",
                "an_3",
                "an_4",
                "an_5",
                "an_6",
            ]
        ]
        .median(axis=1)
        .round()
    )
    .assign(id_psy_anxiety_sympt=lambda df: df.id_psy_anxiety_sympt.astype(cat_012_ord))
    .drop(
        columns=[
            "an_1",
            "an_2",
            "an_3",
            "an_4",
            "an_5",
            "an_6",
        ]
    )
)


# %% jupyter={"outputs_hidden": false}
clean_cols.append("id_psy_anxiety_sympt")
df_survey.id_psy_anxiety_sympt.value_counts()


# %% [markdown]
# ## Household income issues and levels

# %% jupyter={"outputs_hidden": false}
# Problemas con ingresos familiares
#
df_survey = (
    df_survey.assign(
        ef_1=lambda df: np.select(
            [df.ef_1 == 1, df.ef_1 == 2, df.ef_1 == 3, df.ef_1 == 9], [2, 1, 0, np.nan]
        ),
        ef_10=lambda df: np.select(
            [df.ef_10.between(1, 2), df.ef_10 == 3, df.ef_10 == 4, df.ef_10 == 9],
            [2, 1, 0, np.nan],
        ),
        sd_inc_income_level=lambda df: df[["ef_1", "ef_10"]].median(axis=1).round(),
    )
    .assign(sd_inc_income_level=lambda df: df.sd_inc_income_level.astype(cat_012_ord))
    .assign(
        ef_3=lambda df: np.select(
            [df.ef_3 == 1, df.ef_3 == 2, df.ef_3 == 9], [1, 0, np.nan]
        ),
        ef_4=lambda df: np.select(
            [df.ef_4 == 1, df.ef_4 == 2, df.ef_4 == 9], [1, 0, np.nan]
        ),
        ef_5=lambda df: np.select(
            [df.ef_5 == 1, df.ef_5 == 2, df.ef_5 == 9], [1, 0, np.nan]
        ),
        ef_6=lambda df: np.select(
            [df.ef_6 == 1, df.ef_6 == 2, df.ef_6 == 9], [1, 0, np.nan]
        ),
        ef_7=lambda df: np.select(
            [df.ef_7 == 1, df.ef_7 == 2, df.ef_7 == 9], [0, 1, np.nan]
        ),
        ef_8=lambda df: np.select(
            [df.ef_8 == 1, df.ef_8 == 2, df.ef_8 == 9], [0, 1, np.nan]
        ),
        ef_9=lambda df: np.select(
            [df.ef_9 == 1, df.ef_9 == 2, df.ef_9 == 9], [0, 1, np.nan]
        ),
    )
    .assign(
        sd_inc_income_problems=lambda df: df[
            [
                "ef_3",
                "ef_4",
                "ef_5",
                "ef_6",
                "ef_7",
                "ef_8",
                "ef_9",
            ]
        ]
        .median(axis=1)
        .round()
    )
    .assign(sd_inc_income_problems=lambda df: df.sd_inc_income_problems.astype("bool"))
    .drop(
        columns=[
            "ef_1",
            "ef_3",
            "ef_4",
            "ef_5",
            "ef_6",
            "ef_7",
            "ef_8",
            "ef_9",
            "ef_10",
        ]
    )
)


# %% jupyter={"outputs_hidden": false}
clean_cols.append("sd_inc_income_problems")
df_survey.sd_inc_income_problems.value_counts()

# %% jupyter={"outputs_hidden": false}
clean_cols.append("sd_inc_income_level")
df_survey.sd_inc_income_level.value_counts()

# %% [markdown]
# ## Behavioral risk on  Holiday festivities

# %% jupyter={"outputs_hidden": false}
fa_columns = [col for col in df_survey.columns if col.startswith("fa")]

for col in fa_columns:
    df_survey[col] = np.select(
        [
            df_survey[col] == 1,
            df_survey[col] == 2,
            df_survey[col] == 3,
            df_survey[col] > 8,
        ],
        [2, 1, 0, np.nan],
    )

df_survey = (
    df_survey.assign(
        sd_cul_holiday_season=lambda df: df[fa_columns].median(axis=1).round()
    )
    .assign(
        sd_cul_holiday_season=lambda df: df.sd_cul_holiday_season.astype(cat_012_ord)
    )
    .drop(columns=fa_columns)
)


# %% jupyter={"outputs_hidden": false}
clean_cols.append("sd_cul_holiday_season")
df_survey.sd_cul_holiday_season.value_counts()

# %% [markdown]
# ## Gender

# %% jupyter={"outputs_hidden": false}
# Variables Sociodograficas
# Sexo y Edad
df_survey = (
    df_survey.rename(
        columns={
            "cs_1": "id_bio_gender",
        }
    )
    .assign(
        id_bio_gender=lambda df: np.select(
            [df.id_bio_gender == 1, df.id_bio_gender == 2], [1, 0]
        )
    )
    .assign(id_bio_gender=lambda df: df.id_bio_gender.astype("bool"))
)


# %% jupyter={"outputs_hidden": false}
clean_cols.append("id_bio_gender")
df_survey.id_bio_gender.value_counts()

# %% [markdown]
# ## Age

# %% jupyter={"outputs_hidden": false}
df_survey = df_survey.rename(columns={"cs_2": "id_bio_age"}).assign(
    id_bio_age=lambda df: df.id_bio_age.astype("float")
)
# %% jupyter={"outputs_hidden": false}
clean_cols.append("id_bio_age")

plt.hist(df_survey.id_bio_age)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Distribution of Age")
plt.show()

# %% [markdown]
# ## Height and weight

# %% jupyter={"outputs_hidden": false}
df_survey = df_survey.assign(
    cs_3=lambda df: np.where(df.cs_3 >= 98, np.nan, df.cs_3),
    cs_4=lambda df: np.where(df.cs_4 == 999, np.nan, df.cs_4),
)


df_survey = df_survey.assign(
    cs_3=lambda df: df.cs_3.astype("float"),
    cs_4=lambda df: df.cs_4.astype("float"),
).rename(columns={"cs_3": "id_bio_weight", "cs_4": "id_bio_height"})
# %% jupyter={"outputs_hidden": false}
clean_cols.append("id_bio_weight")

counts = df_survey.id_bio_weight
plt.hist(counts)
plt.xlabel("Kilogrames")
plt.ylabel("Frequency")
plt.title("Distribution of Weight")
plt.show()

# %% jupyter={"outputs_hidden": false}
clean_cols.append("id_bio_height")
plt.hist(df_survey.id_bio_height)
plt.xlabel("Meters")
plt.ylabel("Frequency")
plt.title("Distribution of Height")
plt.show()
# %%
aa = np.sort(df_survey.id_bio_weight / (df_survey.id_bio_height / 100) ** 2)

# %%
plt.hist(df_survey.id_bio_weight / (df_survey.id_bio_height / 100) ** 2, bins=20)
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.title("Distribution of BMI")
plt.show()

# %% [markdown]
# ## Education

# %% jupyter={"outputs_hidden": false}
df_survey = df_survey.assign(
    sd_edu_level=lambda df: np.select(
        [
            df.cs_5.between(1, 2),
            df.cs_5.between(3, 4),
            df.cs_5.between(5, 6),
            df.cs_5 == 9,
        ],
        [2, 1, 0, np.nan],
    )
).assign(sd_edu_level=lambda df: df.sd_edu_level.astype(cat_012_ord))


# %% jupyter={"outputs_hidden": false}
clean_cols.append("sd_edu_level")
df_survey.sd_edu_level.value_counts()

# %% [markdown]
# ## Occupation: Working, Studying or Retired

# %% jupyter={"outputs_hidden": false}
# Ocupacion
df_survey = (
    df_survey.assign(
        sd_occ_current_job=lambda df: np.select(
            [
                df.cs_6_a.isin([1, 2, 6, 7]),
                df.cs_6_a.isin([3, 4, 5]),
                df.cs_6_a == 9,
            ],
            [1, 0, np.nan],
        )
    )
    .assign(sd_occ_current_job=lambda df: df.sd_occ_current_job.astype("bool"))
    .drop(columns=["cs_6_a"])
)


# %% jupyter={"outputs_hidden": false}
clean_cols.append("sd_occ_current_job")
df_survey.sd_occ_current_job.value_counts()

# %% [markdown]
# ## Costa Rican origin

# %% jupyter={"outputs_hidden": false}
df_survey = (
    df_survey.assign(
        sd_eth_is_costa_rican=lambda df: np.select(
            [df.cs_7 == 1, df.cs_7 == 2], [True, False]
        )
    )
    .assign(sd_eth_is_costa_rican=lambda df: df.sd_eth_is_costa_rican.astype("bool"))
    .drop(columns=["cs_7"])
)


# %% jupyter={"outputs_hidden": false}
clean_cols.append("sd_eth_is_costa_rican")
df_survey.sd_eth_is_costa_rican.value_counts()

# %% [markdown]
# ## Comorbidities

# %% jupyter={"outputs_hidden": false}
# Comorbilidades
df_survey = (
    df_survey.assign(
        cs_8=lambda df: np.select(
            [df.cs_8 == 1, df.cs_8 == 2, df.cs_8 == 9], [1, 0, np.nan]
        ),
        cs_9=lambda df: np.select(
            [df.cs_9 == 1, df.cs_9 == 9, df.cs_9 == 2], [1, 1, 0]
        ),
    )
    .assign(
        id_bio_comorbidities=lambda df: df[["cs_8", "cs_9"]]
        .sum(axis=1)
        .round()
        # .mode(axis=1, dropna=True)
        #  .max(axis=1)
    )
    .assign(id_bio_comorbidities=lambda df: df.id_bio_comorbidities.astype("bool"))
    .drop(columns=["cs_8", "cs_9"])
)


# %% jupyter={"outputs_hidden": false}
clean_cols.append("id_bio_comorbidities")
df_survey.id_bio_comorbidities.value_counts()

# %%

# %% [markdown]
# ## Religion affiliation

# %% jupyter={"outputs_hidden": false}
# Religion
df_survey = (
    df_survey.rename(columns={"cs_10_br": "sd_cul_religion"})
    .assign(
        sd_cul_religion=lambda df: np.select(
            [df.sd_cul_religion.isin([1, 7]), ~df.sd_cul_religion.isin([1, 7])],
            [1, 0],
        )
    )
    .assign(sd_cul_religion=lambda df: df.sd_cul_religion.astype("bool"))
    .drop(columns=["cs_10_a"])
)

# %% jupyter={"outputs_hidden": false}
clean_cols.append("sd_cul_religion")
df_survey.sd_cul_religion.value_counts()

# %% [markdown]
# ## Disabilities

# %% jupyter={"outputs_hidden": false}
df_survey = (
    df_survey.assign(
        id_bio_disability=lambda df: np.select(
            [df.cs_10_c == 1, df.cs_10_c == 0], [1, 0]
        )
    )
    .assign(id_bio_disability=lambda df: df.id_bio_disability.astype("bool"))
    .drop(columns=["cs_10_c"])
)


# %% jupyter={"outputs_hidden": false}
clean_cols.append("id_bio_disability")
df_survey.id_bio_disability.value_counts()

# %% [markdown]
# ## Total members in household

# %% jupyter={"outputs_hidden": false}
df_survey = df_survey.assign(
    cs_11_a=lambda df: df.cs_11_a.astype("float"),
    cs_11_b=lambda df: df.cs_11_b.astype("float"),
).rename(
    columns={
        "cs_11_a": "id_mat_total_house_members",
        "cs_11_b": "id_mat_18p_house_members",
    }
)

# %%
clean_cols.append("id_mat_18p_house_members")
plt.hist(df_survey.id_mat_18p_house_members)
plt.xlabel("Members")
plt.ylabel("Frequency")
plt.title("Distribution of Members over 18")
plt.show()

# %%
clean_cols.append("id_mat_total_house_members")
plt.hist(df_survey.id_mat_total_house_members)
plt.xlabel("Members")
plt.ylabel("Frequency")
plt.title("Distribution of Total Members")
plt.show()
# %%
plt.hist(
    df_survey.id_mat_18p_house_members / df_survey.id_mat_total_house_members,
    bins=10,
)
plt.xlabel("Members")
plt.ylabel("Frequency")
plt.title("Distribution of Total Members")
plt.show()

# %%
clean_cols

# %% jupyter={"outputs_hidden": false}
df_survey = df_survey[clean_cols]
df_survey.head()

# %%
skimpy.skim(df_survey)

# %% jupyter={"outputs_hidden": false}
df_survey.to_pickle(path="data/df_survey.pkl")
