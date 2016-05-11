package org.trustedanalytics.sparktk.saveload

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.json4s.JsonAST.JValue
import org.json4s.NoTypeHints
import org.json4s.jackson.Serialization

/**
 * Library to save/load json metadata in a tk/ subfolder
 */
object TkSaveLoad {
  /**
   * Helper function for TkSaveable, this function loads TkMetadata from a file with
   * the appropriate path.  spark-tk just adds a tk/ folder alongside the data/ and metadata/ folders
   * saved by spark models.
   *
   * @param sc active spark context
   * @param path the source path
   * @return a tuple of (formatId, formatVersion, data)
   */
  def loadTk(sc: SparkContext, path: String): (String, Int, JValue) = {
    SaveLoad.load(sc, tkMetadataPath(path))
  }

  /**
   * Helper function for inheritors of TkSaveable, this function saves the TkMetadata to a file with
   * the appropriate path.  spark-tk just adds a tk/ folder alongside the data/ and metadata/ folders
   * saved by spark models.
   *
   * @param sc active spark context
   * @param path the destination path
   * @param formatId the identifier of the format type, usually the full name of a case class type
   * @param formatVersion the version of the format for the tk metadata that should be recorded.
   * @param tkMetadata the data to save (should be a case class), must be serializable to JSON using json4s
   */
  def saveTk(sc: SparkContext, path: String, formatId: String, formatVersion: Int, tkMetadata: Any) = {
    SaveLoad.save(sc, tkMetadataPath(path), formatId, formatVersion, tkMetadata)
  }

  def tkMetadataPath(path: String): String = new Path(path, "tk").toUri.toString

  def extract[T <: Product](tkMetaJson: JValue)(implicit t: Manifest[T]): T = {
    implicit val formats = Serialization.formats(NoTypeHints)
    tkMetaJson.extract[T]
  }
}
