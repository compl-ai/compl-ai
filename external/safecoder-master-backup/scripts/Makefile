SRCS=$(wildcard *.c)
OBJS=$(SRCS:.c=.o)

all: $(OBJS)

%.o: %.c
	gcc -c $< -o $@