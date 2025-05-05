"""day_3"""
import pandas as pd


# pd.set_option("display.max_rows", None)
# pd.set_option("display.max_columns",None)

""" To read the CSV """
df = pd.read_csv("project1_df.csv")
# print(df.head())   # first 5 rows
# print(df.tail())   # last 5 rows

""" To read the specific rows"""
# print(df[0:2])  # read 0,1 rows
# print(df.loc[0:2])   # read 0,1,2 rows

"""To read specific rows in specific column using loc"""
# print(df.loc[0:3,["Purchase Method","Location","Net Amount"]])  # loc - access through column names

"""To read specific rows in specific columns using iloc"""
# print(df.iloc[0:3,1:2]) # iloc - access through indices

"""To print the column names"""
# print(df.columns)

"""To print specific column"""
# print(df["Location"].head(10))  # 0 to 9 rows
# print(df["Location"].loc[0:10])  # 0 to 10 rows

# print(df["Location"].dtypes) # datatype of a specific column

# print(df[["Location", "Net Amount"]]) #use double bracket to print 2 or more columns

""" To print the non-null count, dtypes of all columns """
# print(df.info())

"""To print the statistical information of each column"""
# print(df.describe())

""" To print number of rows and columns"""
# print(df.shape)

"""To print the dimension"""
# print(df.ndim)

"""To drop a column or multiple columns as well """
# df.drop(columns = ["Location"], inplace = True)  # or
# df.drop(["Location"], axis = 1, inplace=True)  # or
# df = df.drop(["Location"], axis = 1)
# print(df)

"""day_4"""
"""Finding null in every column using true or false"""
# print(df.isna()) # or
# print(df.isnull())

"""Finding total count of null in every column"""
# print(df.isna().sum())   # or
# print(df.isnull().sum())

"""Finding total null of all columns"""
# print(df.isna().sum().sum())   # or
# print(df.isnull().sum().sum())

"""unique data in particular column"""
# print(df["Product Category"].unique())

"""unique data with its count in particular column"""
# print(df["Product Category"].value_counts())

"""Filling to null data"""
# finding_null = df.isna().sum()
# print(finding_null)

# print("--------------------------------")

"""drop null rows"""
# df.dropna(inplace = True)
# print(df.isna().sum())

"""Forward fill, backword fill, fill na"""
df["Discount Name"]=df["Discount Name"].ffill()
# df.bfill(inplace = True)

# df.fillna(20, inplace = True)
# df["Discount Name"].fillna(1000, inplace = True)
# df["Discount Name"].ffill(inplace = True)
# print(df.isna().sum())


"""Filling null values using SimpleImputer (strategy)"""

from sklearn.impute import SimpleImputer
# pd.set_option("display.max_columns",None)
# print(df.head())

# print(df.iloc[0:10, 8:9])
# print(df["Discount Amount (INR)"].head())

"""fill using simple imputer by the value mean"""
simple_imputer = SimpleImputer(strategy="mean")
df["Discount Amount (INR)"] = simple_imputer.fit_transform(df[["Discount Amount (INR)"]])
# print(df["Discount Amount (INR)"])

"""fill using simple imputer by the value 2.9"""
# fill_by_simple_imputer = SimpleImputer(strategy="constant", fill_value=2.9)
# fill_by_simple_imputer.fit_transform(df[["Discount Amount (INR)"]])
# print(df.iloc[0:3,0:9])

# print(df["Discount Amount (INR)"].var())

"""Label Encoder must be used for target data"""
from sklearn.preprocessing import LabelEncoder

# print(df.info())

# label_encoding_object = LabelEncoder()  # label encoding is used for target data. Here I used it for practice
# df["Gender"] = label_encoding_object.fit_transform(df["Gender"])
# print(df["Gender"].head())

from sklearn.preprocessing import OrdinalEncoder

# Ordinal_encoding_object = OrdinalEncoder()
# df["Gender"]=Ordinal_encoding_object.fit_transform(df[["Gender"]])
# print(df["Gender"])

# print(df.columns)

"""Multiple categorical data columns to numerical columns in a single shot """

category_cols = [ 'Gender', 'Age Group','Product Category', 'Discount Availed',
                  'Discount Name', 'Purchase Method', 'Location']
encoder = OrdinalEncoder()
df[category_cols] = encoder.fit_transform(df[category_cols])
# print(df)
# #
# print(df.info())

# print(df["Gender"].unique())

""" we can custom ordinals using categories"""
# encoder = OrdinalEncoder(categories=[['Male','Female','Other']])
# df["Gender"] = encoder.fit_transform(df[["Gender"]])
# print(df["Gender"])


"""Day 5"""
"""Data scaling ---> using MinMaxScalar, and StandardScalar"""

print(df.columns)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

"""Minmaxscalar = 0 to 1 """

# numerical_cols = ['Gross Amount', 'Net Amount']
# data_scaling = MinMaxScaler()
# df[numerical_cols] = data_scaling.fit_transform(df[numerical_cols])
# print(df[numerical_cols])

"""Standardscalar ====> mean = 0, std = 1"""
# data_scaling = StandardScaler()
# df["Gross Amount"] = data_scaling.fit_transform(df[["Gross Amount"]])
# print(df["Gross Amount"])


"""Data transformation"""  """Using pandas"""

# print(df['Discount Availed'].unique())

"""get dummies splits one column into many columns depends on unique catergories"""
# data = pd.get_dummies(df["Discount Availed"])
# print(data)

"""If we want to add prefix in the column name, we use prefix parameter """
# data = pd.get_dummies(df["Discount Availed"],prefix = "Discount Availed")
# print (data)

"""Converting the above columns into dataframe"""
# convert_to_datafr = pd.DataFrame(data)

"""Now, joining the new dataframe with old df"""
# New_df = pd.concat([df, convert_to_datafr], axis= 1)
# print(New_df.head())

"""Dropping the original columns"""
# New_df.drop("Discount Availed",axis = 1, inplace = True)
# print(New_df.columns)

"""Data transformation >>>> using scikit >>> OneHotEncoder"""

from sklearn.preprocessing import OneHotEncoder

"""To get array of data"""
# one_hot_encoding = OneHotEncoder(sparse_output=False)  # false returns numpy array >> easy to convert dataframe, true returns sparse matrix >> use .toarray()
# data = one_hot_encoding.fit_transform(df[["Discount Availed"]])
# print(data)

"""to get feature names"""
# Encoded_data_columns = one_hot_encoding.get_feature_names_out(["Discount Availed"])
# print(Encoded_data_columns)

"""Conversion to dataframe"""
# Encoded_data_to_dataframe = pd.DataFrame(data, columns = Encoded_data_columns)
# print(Encoded_data_to_dataframe)

"""Concatenation of df with encoded_dataframe"""
# New_df = pd.concat([df, Encoded_data_to_dataframe],axis = 1)
# print(New_df)
#
# print(New_df.columns)

"""Dropping the the old column "Discount Availed" """
# New_df.drop(["Discount Availed"],axis=1, inplace = True)
#
# print(New_df.columns)


"""Day 6"""  """Visualization using EDA"""
