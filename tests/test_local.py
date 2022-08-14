from riverrem.REMMaker import REMMaker

test_dem = './smith_SRTM.tif'

if __name__ == "__main__":
    rem_maker = REMMaker(dem=test_dem)
    rem_maker.make_rem_viz(cmap='Blues', z=10)
