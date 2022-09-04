package main

import (
	"bufio"
	"io"
	"io/ioutil"
	"log"
	"fmt"
	"os"
	"sort"
	"github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

const (
	graphFile  = "model/tensorflow_inception_graph.pb"
	labelsFile = "model/imagenet_comp_graph_label_strings.txt"
)

// Label type
type Label struct {
	Label       string  `json:"label"`
	Probability float32 `json:"probability"`
}

// Labels type
type Labels []Label

func (a Labels) Len() int           { return len(a) }
func (a Labels) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a Labels) Less(i, j int) bool { return a[i].Probability > a[j].Probability }


func getTensorFlowProbability(body io.ReadCloser, modelGraph *tensorflow.Graph,session *tensorflow.Session, labels []string) (string, float32) {
	bytes, err := ioutil.ReadAll(body)
	if err != nil {
		log.Fatal(err)
	}

	// Make tensor
	tensor, err := makeTensorFromImage(bytes)
	if err != nil {
		fmt.Println("makeTensorFromImage error", err)
	}

	// - - - - - - - - - - - - - - - - -

	output, err := session.Run(
		map[tensorflow.Output]*tensorflow.Tensor{
			modelGraph.Operation("input").Output(0): tensor,
		},
		[]tensorflow.Output{
			modelGraph.Operation("output").Output(0),
		},
		nil)
	
	if err != nil {
		log.Fatalf("could not run inference: %v", err)
	}

	// - - - - - - - - - - - - - - - - -

	res := getTopFiveLabels(labels, output[0].Value().([][]float32)[0])
	
	return res[0].Label, res[0].Probability*100
}

func loadModel() (*tensorflow.Graph, []string, error) {
	// Load inception model
	model, err := ioutil.ReadFile(graphFile)
	if err != nil {
		return nil, nil, err
	}
	graph := tensorflow.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		return nil, nil, err
	}

	// Load labels
	labelsFile, err := os.Open(labelsFile)
	if err != nil {
		return nil, nil, err
	}
	defer labelsFile.Close()

	scanner := bufio.NewScanner(labelsFile)
	var labels []string
	for scanner.Scan() {
		labels = append(labels, scanner.Text())
	}

	return graph, labels, scanner.Err()
}

func makeTensorFromImage(imageBuffer []byte) (*tensorflow.Tensor, error) {
	tensor, err := tensorflow.NewTensor(string(imageBuffer))
	if err != nil {
		return nil, err
	}
	graph, input, output, err := makeTransformImageGraph()
	if err != nil {
		return nil, err
	}
	session, err := tensorflow.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	defer session.Close()
	normalized, err := session.Run(
		map[tensorflow.Output]*tensorflow.Tensor{input: tensor},
		[]tensorflow.Output{output},
		nil)
	if err != nil {
		return nil, err
	}
	return normalized[0], nil
}

// Creates a graph to decode, rezise and normalize an image
func makeTransformImageGraph() (graph *tensorflow.Graph, input, output tensorflow.Output, err error) {
	const (
		H, W  = 224, 224
		Mean  = float32(117)
		Scale = float32(1)
	)
	s := op.NewScope()
	input = op.Placeholder(s, tensorflow.String)
	
	// Decode JPEG
	decode := op.DecodeJpeg(s, input, op.DecodeJpegChannels(3))

	// Div and Sub perform (value-Mean)/Scale for each pixel
	output = op.Div(s,
		op.Sub(s,
			// Resize to 224x224 with bilinear interpolation
			op.ResizeBilinear(s,
				// Create a batch containing a single image
				op.ExpandDims(s,
					// Use decoded pixel values
					op.Cast(s, decode, tensorflow.Float),
					op.Const(s.SubScope("make_batch"), int32(0))),
				op.Const(s.SubScope("size"), []int32{H, W})),
			op.Const(s.SubScope("mean"), Mean)),
		op.Const(s.SubScope("scale"), Scale))
	graph, err = s.Finalize()

	return graph, input, output, err
}

func getTopFiveLabels(labels []string, probabilities []float32) []Label {
	var resultLabels []Label
	for i, p := range probabilities {
		if i >= len(labels) {
			break
		}
		resultLabels = append(resultLabels, Label{Label: labels[i], Probability: p})
	}

	sort.Sort(Labels(resultLabels))
	return resultLabels[:5]
}