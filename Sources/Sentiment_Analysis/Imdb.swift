import TensorFlow
import Foundation
import Datasets

public struct Imdb {

    static func downloadImdbDatasetIfNotPresent() -> URL {
        let localURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let dataFolder = DatasetUtilities.downloadResource(
            filename: "aclImdb_v1",
            fileExtension: "tar.gz",
            remoteRoot: URL(string: "https://ai.stanford.edu/~amaas/data/sentiment/")!,
            localStorageDirectory: localURL.appendingPathComponent("data/", isDirectory: true))

        return dataFolder
    }

    public init() {
        let trainPosFiles = try! FileManager.default.contentsOfDirectory(atPath: FileManager.default.currentDirectoryPath + "/data/aclImdb/train/pos")
        let trainNegFiles = try! FileManager.default.contentsOfDirectory(atPath: FileManager.default.currentDirectoryPath + "/data/aclImdb/train/neg")

        var trainText: [String] = []
        var trainLabel: [Int] = []

        let path = FileManager.default.currentDirectoryPath + "/data/aclImdb/train/"
        for files in trainPosFiles[0...1000]{
            let currentPath = path + "pos/" + files
            let text = try! String(contentsOfFile: currentPath, encoding: .utf8)
            trainText.append(text)
            trainLabel.append(1)
        }
        for files in trainNegFiles[0...1000]{
            let currentPath = path + "neg/" + files
            let text = try! String(contentsOfFile: currentPath, encoding: .utf8)
            trainText.append(text)
            trainLabel.append(0)
        }

        print(trainText.count)
        print(trainLabel.count)



    }
}
