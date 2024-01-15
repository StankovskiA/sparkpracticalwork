from pyspark.sql.functions import col, count, when, skewness, kurtosis, udf, concat_ws
from pyspark.sql.types import StringType, NumericType, IntegerType, DoubleType
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.window import Window
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline

def clean_data(df):
    '''
    Remove columns that are not needed or useful for the analysis
    '''
    # List of columns to be removed
    columns_to_remove = ['ArrTime', 'ActualElapsedTime', 'AirTime', 'TaxiIn', 'Diverted',
                        'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']

    # Remove columns with only one unique value
    for c in [c for c in df.columns if c not in columns_to_remove]:
        if df.select(c).distinct().count() == 1:
            print("Column '{}' has only one unique value".format(c))
            columns_to_remove.append(c)
            
    # Select columns that are NOT in the 'columns_to_remove' list
    df = df.select([c for c in df.columns if c not in columns_to_remove])
    
    # Filtering cancelled flight
    # Count the number of canceled flights before filtering
    cancelled_flights_count_before = df.filter(col('Cancelled') == 1).count()
    print(f"Number of cancelled flights before filtering: {cancelled_flights_count_before}")

    # Remove canceled flights
    df_not_cancelled = df.filter(col('Cancelled') == 0)

    # Count the number of canceled flights after filtering
    cancelled_flights_count_after = df_not_cancelled.filter(col('Cancelled') == 1).count()
    print(f"Number of canceled flights after filtering: {cancelled_flights_count_after}")

    # Drop the 'Cancelled' column from the DataFrame
    df = df_not_cancelled.drop('Cancelled')

    return df

def columns_by_type(df):
    '''
    Returns lists with the columns of the DataFrame grouped by type
    '''
    target_var = 'ArrDelay'
    categorical_cols = ['UniqueCarrier', 'Origin', 'Dest']
    numerical_cols = [x for x in df.columns if x not in categorical_cols]

    # Remove the target variable 'ArrDelay' from the lists
    if target_var in numerical_cols:
        numerical_cols.remove(target_var)
    if target_var in categorical_cols:
        categorical_cols.remove(target_var)
        
    # Print the lists
    print("Numerical Columns:", numerical_cols)
    print("Categorical Columns:", categorical_cols)

    return categorical_cols, numerical_cols

def encode_categorical_columns(df, categorical_cols):
    '''
    Label encode categorical columns
    '''
    print(f"Label encoding categorical columns {categorical_cols}")
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_encoded") for c in categorical_cols]
    pipeline = Pipeline(stages=indexers)
    df_encoded = pipeline.fit(df).transform(df)

    # Drop the original categorical columns
    df_encoded = df_encoded.drop(*categorical_cols)
    return df_encoded

def data_statistics(df_encoded, numerical_cols, categorical_cols):
    '''
    Perform data exploration on the DataFrame
    '''
    # Calculating summary statistics for numerical columns
    print("Summary statistics for numerical columns :")
    df_encoded.describe(numerical_cols).show()

    # Analyzing frequency counts for encoded categorical columns
    encoded_categorical_cols = [f"{c}_encoded" for c in categorical_cols]
    for col_name in encoded_categorical_cols:
        print(f"Frequency counts for {col_name} :")
        df_encoded.groupBy(col_name).count().orderBy('count', ascending=False).show()

    # Checking for missing values in each column
    print("Missing values in each column:")
    df_encoded.select([count(when(col(c).isNull(), c)).alias(c) for c in df_encoded.columns]).show()
    
    # Calculate skewness for numerical columns
    for c in numerical_cols:
        skewness_value = df_encoded.select(skewness(c)).collect()[0][0]
        print(f"Skewness of {c}: {skewness_value}")

    # Calculate kurtosis for numerical columns
    for c in numerical_cols:
        kurtosis_value = df_encoded.select(kurtosis(c)).collect()[0][0]
        print(f"Kurtosis of {c}: {kurtosis_value}")
        
    # Filter out non-numerical columns based on the actual data type
    actual_numerical_cols = [f.name for f in df_encoded.schema.fields if isinstance(f.dataType, NumericType)]

    # Define the bounds for the IQR
    bounds = {
        c: dict(
            zip(["q1", "q3"], df_encoded.approxQuantile(c, [0.25, 0.75], 0))
        )
        for c in actual_numerical_cols
    }

    for c in actual_numerical_cols:
        iqr = bounds[c]['q3'] - bounds[c]['q1']
        lower_bound = bounds[c]['q1'] - (1.5 * iqr)
        upper_bound = bounds[c]['q3'] + (1.5 * iqr)
        
        print(f"Column {c}:")
        print(f"    Lower bound: {lower_bound}")
        print(f"    Upper bound: {upper_bound}")
        
        # Optional: Filter out the outliers from the DataFrame
        df_no_outliers = df_encoded.filter((col(c) >= lower_bound) & (col(c) <= upper_bound))
        
        # Optional: View the count of identified outliers
        outliers_count = df_encoded.filter((col(c) < lower_bound) | (col(c) > upper_bound)).count()
        print(f"    Identified outliers: {outliers_count}")

def process_datetimes(df_encoded):
    '''
    Create new features based on the datetime columns
    '''
    def get_part_of_day(deptime):
        ''' 
        UDF to classify the time of day based on DepTime in hhmm format
        '''
        if deptime is None:
            return None
        hour = int(deptime) // 100
        if 5 <= hour <= 11:
            return 'Morning'
        elif 12 <= hour <= 17:
            return 'Afternoon'
        elif 18 <= hour <= 22:
            return 'Evening'
        else:
            return 'Night'
          
    part_of_day_udf = udf(get_part_of_day, StringType())

    # Now, let's recast the DepTime column to IntegerType to handle cases where it's not an integer.
    df_encoded = df_encoded.withColumn('DepTime', col('DepTime').cast(IntegerType()))

    # Then use the UDF to create the PartOfDay column
    df_encoded = df_encoded.withColumn('PartOfDay', part_of_day_udf(col('DepTime')))

    # Add IsWeekend column
    df_encoded = df_encoded.withColumn('IsWeekend', when(col('DayOfWeek').isin([6, 7]), 1).otherwise(0))

    # Add Season column based on the month
    df_encoded = df_encoded.withColumn('Season', when(col('Month').isin([12, 1, 2]), 'Winter')
                                        .when(col('Month').isin([3, 4, 5]), 'Spring')
                                        .when(col('Month').isin([6, 7, 8]), 'Summer')
                                        .otherwise('Autumn'))

    # Create a 'FlightDate' key using 'Month' and 'DayofMonth' (as we only have data for 1987)
    df_encoded = df_encoded.withColumn('FlightDate', 
                                    concat_ws("-", col('Month').cast(StringType()), col('DayofMonth').cast(StringType())))
    
    return df_encoded

def encode_datecat_columns(df_encoded):
    '''
    Encode the categorical columns using OneHotEncoder
    '''
    print("Encoding date categorical columns using OneHotEncoder")
    # Define the columns to be indexed and encoded
    categorical_cols_to_encode = ['Season', 'PartOfDay']

    # Create a list to hold the stages of the pipeline
    stages = []

    # Iterate over the columns to create indexing and encoding stages
    for categorical_col in categorical_cols_to_encode:
        # Create a StringIndexer
        string_indexer = StringIndexer(inputCol=categorical_col, outputCol=categorical_col + "Index")
        
        # Create a OneHotEncoder
        encoder = OneHotEncoder(inputCols=[string_indexer.getOutputCol()], outputCols=[categorical_col + "Vec"])
        
        # Add the indexers and encoders to our pipeline stages
        stages += [string_indexer, encoder]

    # Create the pipeline
    pipeline = Pipeline(stages=stages)

    # Fit the pipeline to the data
    pipeline_model = pipeline.fit(df_encoded)

    # Transform the data
    df_encoded = pipeline_model.transform(df_encoded)

    # Now that we have our encoded features, we can remove the original categorical columns
    df_encoded = df_encoded.drop(*categorical_cols_to_encode)

    # And we can also remove the intermediate index columns
    for categorical_col in categorical_cols_to_encode:
        df_encoded = df_encoded.drop(categorical_col + "Index")

    # Show the DataFrame with the new encoded columns
    df_encoded.show(truncate=False)
    
    return df_encoded

def handle_missing_data(df_encoded):
    '''
    Handle missing data in the DataFrame
    '''
    print("Handling missing data")
    # Generate a check for each column if there is any null value
    null_checks = [count(when(col(c).isNull(), c)).alias(c) for c in df_encoded.columns]

    # Apply the checks to the DataFrame
    null_counts = df_encoded.select(*null_checks).collect()[0].asDict()

    # Print out the counts of nulls for each column
    for column, null_count in null_counts.items():
        if null_count != 0: 
            print(f"Column {column} has {null_count} null values")
            
    total_lines = df_encoded.count()
    print(f"The total number of lines in the DataFrame is: {total_lines}")
    
    # List of columns to check for null values
    columns_to_check = ['ArrDelay', 'Distance']

    # Drop rows that have null values in the specified columns
    df_encoded = df_encoded.na.drop(subset=columns_to_check)

    new_total_lines = df_encoded.count()

    # Show the number of lines remaining after removing rows with nulls in specific columns
    print(f"The number of lines after removing rows with null values in specified columns: {new_total_lines}")
    print(f"\nTotal number of deleted lines : {total_lines - new_total_lines}")
    
    return df_encoded

def split_data(df_encoded):
    '''
    Split the data into training and testing sets
    '''
    # Define the features and target variable
    feature_cols = ['Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime', 'CRSArrTime', 'FlightNum',
                    'CRSElapsedTime', 'DepDelay', 'Distance', 'UniqueCarrier_encoded', 'Origin_encoded',
                    'Dest_encoded', 'PartOfDayVec', 'IsWeekend', 'SeasonVec', 'DailyFlightVolume']

    target_col = 'ArrDelay'

    # Assemble features into a single vector column
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="skip"  # Skip lines with null values
    )

    df_assembled = assembler.transform(df_encoded)

    # Select only the features and target variable
    df_model_data = df_assembled.select(col("features"), col(target_col).alias("label"))

    # Split the data into training and testing sets
    train_data, test_data = df_model_data.randomSplit([0.8, 0.2], seed=42)

    return train_data, test_data

def fit_model(train_data):
    '''
    Fit the model to the training data
    '''
    # Initialize the regression model
    rf = RandomForestRegressor(featuresCol="features", labelCol="label", maxBins=250)

    # Train the model
    rf_model = rf.fit(train_data)

    return rf_model

def predict_eval(model, test_data, save_model=True):
    # Make predictions on the test data
    predictions = model.transform(test_data)
    
    # Evaluate the model
    def eval_model(metrics):
        for m in metrics:
            evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName=m)
            metric = evaluator.evaluate(predictions)
            print(f" {m} on test data: {metric}")

    performance_metrics = ['rmse', 'mae', 'mse', 'r2']
    eval_model(performance_metrics)

    if save_model:
        # Save the trained model for later use
        model.write().overwrite().save("model")

def main():
    # Create a Spark session
    spark = SparkSession.builder.appName("SparkMLApp").getOrCreate()

    # Get the input data location from the command line or configuration
    input_data_location = "data/1987.csv"

    print("Reading and cleaning the data")
    
    # Load the data into a PySpark DataFrame and clean it
    df = clean_data(spark.read.csv(input_data_location, header=True, inferSchema=True))

    print("Begin Data Exploration")

    categorical_cols, numerical_cols = columns_by_type(df)
    
    # Label encode categorical columns
    df_encoded = encode_categorical_columns(df, categorical_cols)
    
    # Show the DataFrame with label-encoded categorical columns
    print(f"DataFrame with label-encoded categorical columns:")
    df_encoded.show()

    data_statistics(df_encoded, numerical_cols, categorical_cols)
    
    print("Begin Feature Engineering")

    df_encoded = process_datetimes(df_encoded)
    
    # Window for calculating the daily flight volume
    daily_volume_window = Window.partitionBy('FlightDate')

    # Add DailyFlightVolume column (counting the number of flights per day)
    df_encoded = df_encoded.withColumn('DailyFlightVolume', count('*').over(daily_volume_window))

    # Show the DataFrame with new features
    df_encoded.select('FlightDate', 'PartOfDay', 'IsWeekend', 'Season', 'DailyFlightVolume').show(truncate=False)
    
    # Convert columns to the correct data types
    df_encoded = df_encoded.withColumn("DepDelay", col("DepDelay").cast(DoubleType()))
    df_encoded = df_encoded.withColumn("Distance", col("Distance").cast(DoubleType()))
    df_encoded = df_encoded.withColumn('ArrDelay', col('ArrDelay').cast(DoubleType()))

    df_encoded = encode_datecat_columns(df_encoded)
    df_encoded = handle_missing_data(df_encoded)
    
    print("Begin Model Training and Evaluation")
    train_data, test_data = split_data(df_encoded)
    
    model = fit_model(train_data)
    predict_eval(model, test_data)
    
    # Finish Session
    spark.stop()

if __name__ == "__main__":
    main()