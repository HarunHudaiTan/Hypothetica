.PHONY: all clean distclean watch paper paper-watch paper-clean paper-distclean docker docker-watch docker-clean docker-distclean

all:
	LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 latexmk

watch:
	LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 latexmk -pvc

clean:
	LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 latexmk -c

distclean:
	LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 latexmk -C

paper:
	docker compose run --rm latex

paper-watch:
	docker compose run --rm latex latexmk -pvc

paper-clean:
	docker compose run --rm latex latexmk -c

paper-distclean:
	docker compose run --rm latex latexmk -C

docker: paper

docker-watch: paper-watch

docker-clean: paper-clean

docker-distclean: paper-distclean
