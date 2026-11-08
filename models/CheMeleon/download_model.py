from urllib.request import urlretrieve

url = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"

urlretrieve(url, "chemeleon_mp.pt")

print("CheMeleon model downloaded successfully!")
