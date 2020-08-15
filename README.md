# How to build and use the devito images

## At the devito directory

To build a new image and erase the old 'none' tagged images:
```ruby
sudo docker-compose build devito
sudo docker rmi $(sudo docker images -f “dangling=true” -q)
```

To run the jupyter notebook server:
```ruby
sudo docker-compose up devito
```

To open the container:
```ruby
sudo docker-compose run devito /bin/bash
```
## Observation

### Changes in the container don't imply changes in the local files, so don't forget to download your changes to make sure you don't lose your changes.