#
# Run make -f make.skysub all
#
# $Id: make.skysub 945 2011-03-09 21:29:50Z mb144 $
#
NAME = regularization

INPUTTEX = 

INPUTFIG = figures/scattered-data.png \
	figures/scattered-shepard.png \
	figures/scattered-shepard-covar.png \
	figures/scattered-regularized-T-noisy.png \
	figures/scattered-regularized-T-noiseless.png \
	figures/scattered-regularized-T-covar.png \
	figures/scattered-unregularized-noisy.png \
	figures/scattered-unregularized-noiseless.png \
	figures/scattered-unregularized-covar.png

all: $(NAME).pdf

$(NAME).pdf: $(NAME).tex $(INPUTTEX) $(INPUTFIG)
	pdflatex $(NAME); bibtex $(NAME); pdflatex $(NAME); pdflatex $(NAME)

clean:
	rm -f *.pdf *.dvi

fullclean:
	rm -f *.pdf *.log *.dvi *.aux *.bbl *.blg

dummy:
