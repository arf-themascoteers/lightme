import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from s2_bands import S2Bands
import numpy as np
import os


class CSVProcessor:
    @staticmethod
    def get_neighbour_columns():
        base = S2Bands.get_all_bands()
        cols = []
        for i in range(8):
            for b in base:
                cols.append(f"{b}_{i}")

        return cols

    @staticmethod
    def aggregate(complete, ag):
        df = pd.read_csv(complete)
        df.drop(columns=CSVProcessor.get_geo_columns(), axis=1, inplace=True)
        spatial_columns = CSVProcessor.get_spatial_columns(df)
        columns_to_agg = df.columns.drop(spatial_columns)

        agg_dict = {}
        agg_dict["counter"] = ("som", 'count')
        agg_dict["som_std"] = ("som", 'std')
        for col in columns_to_agg:
            agg_dict[col] = (col, "mean")

        df_group_object = df.groupby(spatial_columns)
        df_mean = df_group_object.agg(**agg_dict).reset_index()
        df_mean.insert(0, "cell", df_mean.index)
        df_mean = df_mean[df_mean["counter"] >= 1]
        df_mean = df_mean.round(4)
        df_mean.to_csv(ag, index=False)

    @staticmethod
    def make_ml_ready(ag, ml):
        df = pd.read_csv(ag)
        df = CSVProcessor.make_ml_ready_df(df)
        df = df.round(4)
        df.to_csv(ml, index=False)

    @staticmethod
    def make_ml_ready_df(df):
        for col in ["when"]:
            if col in df.columns:
                df.drop(inplace=True, columns=[col], axis=1)
        for col in df.columns:
            if col not in ["scene","row","column","counter","som_std","cell","row_offset","column_offset"]:
                scaler = MinMaxScaler()
                df[col] = scaler.fit_transform(df[[col]])
        return df

    @staticmethod
    def get_spatial_columns(df):
        spatial_columns = ["row", "column"]
        if "scene" in df.columns:
            spatial_columns = ["scene"] + spatial_columns
        return spatial_columns

    @staticmethod
    def get_geo_columns():
        return ["lon", "lat", "when"]

    @staticmethod
    def get_grid_columns():
        non_band_columns = ["scene","cell","row","column","counter","som_std","elevation","moisture","temp","som"]
        band_columns = []
        for i in range(9):
            band_columns.append(f"row_offset_{i}")
            band_columns.append(f"column_offset_{i}")
            for band in S2Bands.get_all_bands():
                band_columns.append(f"{band}_{i}")
        return non_band_columns, band_columns

    @staticmethod
    def gridify(ag, grid, neighbours_root):
        non_band_columns, band_columns = CSVProcessor.get_grid_columns()
        df = pd.read_csv(ag)
        scene_fusion = ("scene" in df.columns)
        dest = None
        for index, row in df.iterrows():
            neighbours, np_neighbours = CSVProcessor.get_neighbours(df, row)
            if neighbours is None or len(neighbours) != 9:
                continue

            new_row = {}

            for c in non_band_columns:
                if not scene_fusion and c == "scene":
                    continue
                new_row[c] = row[c]

            for ind, (i, neighbour) in enumerate(neighbours.iterrows()):
                new_row[f"row_offset_{ind}"] = neighbour["row_offset"]
                new_row[f"column_offset_{ind}"] = neighbour["column_offset"]
                for band in S2Bands.get_all_bands():
                    new_row[f"{band}_{ind}"] = neighbour[band]
            new_df = pd.DataFrame(columns=non_band_columns+band_columns, data=[new_row])
            if dest is None:
                dest = new_df
            else:
                dest = pd.concat((dest, new_df), axis=0)

            file_name = f"{int(row['row'])}_{int(row['column'])}.npy"
            if scene_fusion:
                file_name = f"{int(row['scene'])}_{file_name}"
            np_file = os.path.join(neighbours_root,file_name)
            np.save(np_file, np_neighbours)

        dest = dest.round(4)
        dest.to_csv(grid, index=False)

    @staticmethod
    def get_neighbours(df, row):
        band_columns = S2Bands.get_all_bands()
        the_row = row["row"]
        the_column = row["column"]
        the_scene = None
        scene_fusion = ("scene" in df.columns)
        if scene_fusion:
            the_scene = row["scene"]

        neighbours = None
        np_neighbours = np.zeros((12,3,3))
        row_offset = [-1,0,1]
        col_offset = [-1,0,1]
        for ro in row_offset:
            for co in col_offset:
                target_row = the_row + ro
                target_col = the_column + co
                if scene_fusion:
                    filter = df[(df["row"] == target_row) & (df["column"] == target_col) & (df["scene"] == the_scene)]
                else:
                    filter = df[(df["row"] == target_row) & (df["column"] == target_col)]
                if len(filter) == 0:
                    return None, None
                filter = filter.iloc[0:1]
                filter.insert(0,"column_offset",co)
                filter.insert(0,"row_offset",ro)
                bands = filter[band_columns].to_numpy()
                np_neighbours[:,ro+1, co+1] = bands
                if neighbours is None:
                    neighbours = filter
                else:
                    neighbours = pd.concat((neighbours, filter), axis=0)

        return neighbours, np_neighbours