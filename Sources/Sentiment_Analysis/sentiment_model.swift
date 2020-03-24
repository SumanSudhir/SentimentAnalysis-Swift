import TensorFlow
import Foundation


public struct Sentiment: Module{
    public typealias Scalar = Float
    // Dimension of the Embedding Layer
    @noDerivative public let embDim: Int
    // Number of the most frequent tokens to be used
    @noDerivative public let numWords: Int
    // Number of words in each sentence
    @noDerivative public let sentLen: Int
    // Dimension of hidden Embedding Layer
    @noDerivative public let hidDim: Int
    // NUmber of the CNN layer filters
    @noDerivative public let classDim: Int
    //
    @noDerivative public let dropoutRate: Double

    public var embedLayer: Embedding<Scalar>
    public var conv1: Conv2D<Scalar>
    public var conv2: Conv2D<Scalar>
    public var conv3: Conv2D<Scalar>
    public var output: Dense<Scalar>
    public var dropout: Dropout<Scalar>

    public init(
        embDim: Int,
        numWords: Int,
        sentLen: Int,
        hidDim: Int,
        classDim: Int,
        dropoutRate: Double
    ) {
        self.embDim = embDim
        self.numWords = numWords
        self.sentLen = sentLen
        self.hidDim = hidDim
        self.classDim = classDim
        self.dropoutRate = dropoutRate

        embedLayer = Embedding(vocabularySize: self.numWords, embeddingSize: self.embDim)
        conv1 = Conv2D(filterShape: (3, self.embDim, 1, self.hidDim), padding: .same, activation: relu)
        conv2 = Conv2D(filterShape: (4, self.embDim, 1, self.hidDim), padding: .same, activation: relu)
        conv3 = Conv2D(filterShape: (5, self.embDim, 1, self.hidDim), padding: .same, activation: relu)
        // TODO: Add MaxPool1D and Dropout
        // var pool1 = MaxPool1D<Float>(poolSize: , strides: (2, 2))

        output = Dense(inputSize: 3*self.numWords, outputSize: self.classDim, activation: softmax)
        dropout = Dropout(probability: dropoutRate)
    }
    @differentiable
    public func callAsFunction(_ input: Tensor<Int32>) -> Tensor<Scalar>{

        let embedded = embedLayer(input)
        let conved1 = conv1(embedded)
        let conved2 = conv2(embedded)
        let conved3 = conv3(embedded)

        var concat = conved1.concatenated(with:conved2,alongAxis:-1)
        concat = concat.concatenated(with:conved3,alongAxis:-1)

        return output(concat)
    }
}
