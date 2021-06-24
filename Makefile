PROJDIR := $(realpath $(CURDIR))
SOURCEDIR := $(PROJDIR)/spec

all: callgrind.out.$(shell ps -o ppid $$$$)
training: $(SOURCEDIR)/hog_training.cpp
	g++ $(SOURCEDIR)/hog_training.cpp -o training `pkg-config --cflags --libs opencv`
	$(PROJDIR)/./training $(trainingData)
accuracy: $(SOURCEDIR)/hog_accuracy.cpp
	  g++ $(SOURCEDIR)/hog_accuracy.cpp -o accuracy `pkg-config --cflags --libs opencv`
callgrind_accuracy: accuracy
	valgrind --tool=callgrind ./accuracy $(testPath)
	calgrind_annotate callgrind.out.$(shell ps -o ppid $$$$)
hog_f: $(SOURCEDIR)/hog_test.cpp
	g++ -O3 -g $(SOURCEDIR)/hog_test.cpp -o hog_f `pkg-config --cflags --libs opencv`
callgrind.out.$(shell ps -o ppid $$$$): hog_f
	valgrind --tool=callgrind ./hog_f $(path)
	calgrind_annotate callgrind.out.$(shell ps -o ppid $$$$)
.PHONY: clean
clean:
	rm training
	rm accuracy
	rm callgrind_accuracy
	rm hog_f
	rm callgrind.out.$(shell ps -o ppid $$$$)
