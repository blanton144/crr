#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pybind11/pybind11.h> // must be first
#include <pybind11/numpy.h> // must be first

namespace py = pybind11;

/*
 * sigma.c
 *
 * Simple guess at the sky sigma
 *
 * Mike Blanton
 * 1/2006 */

#define PI 3.14159265358979

#define FREEVEC(a) {if((a)!=NULL) free((char *) (a)); (a)=NULL;}
#define FREEALL FREEVEC(sel);FREEVEC(isel)


#define M 64
#define BIG 1.0e30

static float *diff=NULL;


float dselip(unsigned long k, unsigned long n, float *arr)
{
	void dshell(unsigned long n, float *a);
	unsigned long i,j,jl,jm,ju,kk,mm,nlo,nxtmm,*isel;
	float ahi,alo,sum,*sel;

	if (k < 1 || k > n || n <= 0) {
    printf("bad input to selip");
    exit(1);
  }
	isel=(unsigned long *) malloc(sizeof(unsigned long)*(M+2));
	sel=(float *) malloc(sizeof(float)*(M+2));
	kk=k+1;
	ahi=BIG;
	alo = -BIG;
	for (;;) {
		mm=nlo=0;
		sum=0.0;
		nxtmm=M+1;
		for (i=1;i<=n;i++) {
			if (arr[i-1] >= alo && arr[i-1] <= ahi) {
				mm++;
				if (arr[i-1] == alo) nlo++;
				if (mm <= M) sel[mm-1]=arr[i-1];
				else if (mm == nxtmm) {
					nxtmm=mm+mm/M;
					sel[1 + ((i+mm+kk) % M) - 1]=arr[i-1];
				}
				sum += arr[i-1];
			}
		}
		if (kk <= nlo) {
			FREEALL
			return alo;
		}
		else if (mm <= M) {
			dshell(mm,sel);
			ahi = sel[kk-1];
			FREEALL
			return ahi;
		}
		sel[M+1-1]=sum/mm;
		dshell(M+1,sel);
		sel[M+2-1]=ahi;
		for (j=1;j<=M+2;j++) isel[j-1]=0;
		for (i=1;i<=n;i++) {
			if (arr[i-1] >= alo && arr[i-1] <= ahi) {
				jl=0;
				ju=M+2;
				while (ju-jl > 1) {
					jm=(ju+jl)/2;
					if (arr[i-1] >= sel[jm-1]) jl=jm;
					else ju=jm;
				}
				isel[ju-1]++;
			}
		}
		j=1;
		while (kk > isel[j-1]) {
			alo=sel[j-1];
			kk -= isel[j-1];
      j++;
		}
		ahi=sel[j-1];
	}
}
#undef M
#undef BIG
#undef FREEALL

void dshell(unsigned long n, float *a)
{
	unsigned long i,j,inc;
	float v;
	inc=1;
	do {
		inc *= 3;
		inc++;
	} while (inc <= n);
	do {
		inc /= 3;
		for (i=inc+1;i<=n;i++) {
			v=a[i-1];
			j=i;
			while (a[j-inc-1] > v) {
				a[j-1]=a[j-inc-1];
				j -= inc;
				if (j <= inc) break;
			}
			a[j-1]=v;
		}
	} while (inc > 1);
}

float sigma(py::array_t<float> image_in, 
						int nx,
						int ny,
						int sp)
{
	float tot, sigval;
  int i,j,dx,dy, ndiff;

	float *image = (float *) image_in.request().ptr;

	if(nx==1 && ny==1) {
		return(0.);
	}

	dx=50;
	if(dx>nx/4) dx=nx/4;
	if(dx<=0) dx=1;

	dy=50;
	if(dy>ny/4) dy=ny/4;
	if(dy<=0) dy=1;
	
	diff=(float *) malloc(2*nx*ny*sizeof(float));
	ndiff=0;
	for(j=0;j<ny;j+=dy) {
		for(i=0;i<nx;i+=dx) {
			if(i<nx-sp) {
				diff[ndiff]=fabs(image[i+j*nx]-image[i+sp+j*nx]);
				ndiff++;
			}
			if(j<ny-sp) {
				diff[ndiff]=fabs(image[i+j*nx]-image[i+(j+sp)*nx]);
				ndiff++;
			}
		}
	}

	if(ndiff<=1) {
		return(0.);
	}

	if(ndiff<=10) {
		tot=0.;
		for(i=0;i<ndiff;i++)
			tot+=diff[i]*diff[i];
		return(sqrt(tot/(float) ndiff));
	}

	sigval=(dselip((int) floor(ndiff*0.68),ndiff,diff))/sqrt(2.);
	
	FREEVEC(diff);

	return(sigval);
} /* end dsigma */
