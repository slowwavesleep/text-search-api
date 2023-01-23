### Build
```shell
docker build -t textsearch .
```

### Run
This mounts a folder with images with removed background to the container. Assuming that the folder is in the same
directory, but that can be easily changed.
It should take about a minute to build the index over the provided images on a cpu (although that depends on the number
of images in the mounted folder).
```shell
docker run -d -p 5000:5000 -v "$(pwd)"/no_bg:/textsearch/no_bg textsearch
```

### Test
About 5 seconds on a cpu.
```shell
curl -XPOST -H'content-type: application/json' -d'{"text": "pastel coloured tunic", "n": 5}' http://localhost:5000
```